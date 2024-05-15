use std::process;

use clap::{CommandFactory, Parser};
use stablediffusion::model::autoencoder::{Decoder, DecoderConfig};
use stablediffusion::model::autoencoder::{Encoder, EncoderConfig};
use stablediffusion::model::clip::{CLIPConfig, CLIP};
use stablediffusion::model::stablediffusion::{
    offset_cosine_schedule_cumprod, Diffuser, DiffuserConfig, Embedder, EmbedderConfig,
    LatentDecoder, LatentDecoderConfig, RawImages, RESOLUTIONS,
};
use stablediffusion::model::unet::{UNet, UNetConfig};

use stablediffusion::backend::QKVBackend as Backend;

use burn::{
    config::Config,
    module::{Module, Param},
    nn,
    tensor::{self, Tensor},
};

use anyhow::{anyhow, Context, Result};

use stablediffusion::token::{clip::ClipTokenizer, open_clip::OpenClipTokenizer, Tokenizer};

use burn::tensor::ElementConversion;
use num_traits::cast::ToPrimitive;
use stablediffusion::model::stablediffusion::Conditioning;

use stablediffusion::with_backend::*;

use burn::tensor::Bool;

use std::path::{Path, PathBuf};

use tracing::{debug, error, info, trace, warn};

#[cfg(feature = "torch-backend")]
mod torch_backend {
    pub use burn_tch::{LibTorch, LibTorchDevice};
}

#[cfg(feature = "wgpu-backend")]
mod wgpu_backend {
    //pub use burn_wgpu::Wgpu as Backend;
}

use burn::record::{HalfPrecisionSettings, NamedMpkFileRecorder, Recorder};

fn load_embedder_model<B: Backend>(model_dir: &Path, device: &B::Device) -> Result<Embedder<B>> {
    let config = EmbedderConfig::load(model_dir.join("embedder.cfg"))?;
    tracing::debug!("Finished loading embedder config");
    let record = NamedMpkFileRecorder::<HalfPrecisionSettings>::new()
        .load(model_dir.join("embedder.mpk"), device)?;
    tracing::debug!("Finished loading embedder record");
    let m = config.init(device).load_record(record);
    tracing::debug!("Finished initialising embedder");
    Ok(m)
}

fn load_diffuser_model<B: Backend>(model_dir: &Path, device: &B::Device) -> Result<Diffuser<B>> {
    let config = DiffuserConfig::load(model_dir.join("diffuser.cfg"))?;
    tracing::debug!("Finished loading diffuser config");
    let record = NamedMpkFileRecorder::<HalfPrecisionSettings>::new()
        .load(model_dir.join("diffuser.mpk"), device)?;
    tracing::debug!("Finished loading diffuser record");
    //let record = NamedMpkFileRecorder::<HalfPrecisionSettings>::new().load(model_dir.into(), device)?;
    let m = config.init(device).load_record(record);
    tracing::debug!("Finished initialising diffuser");
    Ok(m)
}

fn load_latent_decoder_model<B: Backend>(
    model_dir: &Path,
    device: &B::Device,
) -> Result<LatentDecoder<B>> {
    let config = LatentDecoderConfig::load(model_dir.join("latent_decoder.cfg"))?;
    tracing::debug!("Finished loading diffuser config");
    let record =
        NamedMpkFileRecorder::<HalfPrecisionSettings>::new().load(model_dir.into(), device)?;
    tracing::debug!("Finished loading diffuser record");
    let m = config.init(device).load_record(record);
    tracing::debug!("Finished initialising latent decoder");
    Ok(m)
}

fn arb_tensor<B: Backend, const D: usize>(dims: [usize; D], device: &B::Device) -> Tensor<B, D> {
    let prod: usize = dims.iter().cloned().product();
    Tensor::arange(0..prod as i64, device)
        .float()
        .sin()
        .reshape(dims)
}

#[derive(Debug, clap::Parser)]
struct Opts {
    /// Directory of the model weights
    #[arg(short = 'd', long)]
    model_dir: PathBuf,

    /// Use the refiner model?
    #[arg(short = 'r', long)]
    use_refiner: bool,

    /// Path of the reference image for inpainting
    #[arg(short = 'i', long)]
    img: Option<PathBuf>,

    /// Left-most pixel of the crop window
    #[arg(long)]
    crop_left: Option<usize>,

    /// Right-most pixel of the crop window
    #[arg(long)]
    crop_right: Option<usize>,

    /// Top-most pixel of the crop window
    #[arg(long)]
    crop_top: Option<usize>,

    /// Bottom-most pixel of the crop window
    #[arg(long)]
    crop_bottom: Option<usize>,

    /// Crop outside or inside the specified crop window?
    #[arg(long)]
    crop_out: bool,

    /// Controls the strength of the adherence to the prompt
    #[arg(short = 'g', long, default_value = "7.5")]
    unconditional_guidance_scale: f64,

    /// Number of diffusion iterations used for generating the image
    #[arg(short = 'n', long, default_value = "30")]
    n_diffusion_steps: usize,

    #[arg(short = 'p', long)]
    prompt: String,

    /// Directory of the image outputs
    #[arg(short = 'o', long)]
    output_dir: PathBuf,
}

struct InpaintingTensors<B: Backend> {
    orig_dims: (usize, usize),
    reference_latent: Tensor<B, 4>,
    mask: Tensor<B, 4, Bool>,
}

fn main() {
    let env_filter = tracing_subscriber::filter::EnvFilter::builder()
        .with_default_directive("sample=DEBUG".parse().unwrap())
        .from_env_lossy();
    tracing_subscriber::fmt::fmt()
        .with_env_filter(env_filter)
        .init();

    let device = burn_wgpu::WgpuDevice::BestAvailable;

    burn_wgpu::init_sync::<burn_wgpu::AutoGraphicsApi>(
        &device,
        burn_wgpu::RuntimeOptions::default(),
    );

    let opts = Opts::parse();

    let inpainting_info = opts.img.map(|ref_dir| {
        debug!("Loading inpainting images");
        let imgs = load_images(&[ref_dir.to_str().unwrap().into()]).unwrap();

        if !RESOLUTIONS
            .iter()
            .any(|&[h, w]| h as usize == imgs.height && w as usize == imgs.width)
        {
            info!("Reference image dimensions are incompatible.\nThe compatible dimensions are:");
            for [h, w] in RESOLUTIONS {
                info!("Width: {}, Height: {}", w, h);
            }
            process::exit(1);
        }

        let crop_left = opts.crop_left.unwrap_or(0);
        let crop_right = opts.crop_right.unwrap_or(imgs.width);
        let crop_top = opts.crop_top.unwrap_or(0);
        let crop_bottom = opts.crop_bottom.unwrap_or(imgs.height);

        assert!(
            crop_right <= imgs.width && crop_bottom <= imgs.height && crop_left < crop_right
                || crop_top < crop_bottom,
            "Invalid crop parameters."
        );

        // compute latent
        info!("Loading latent encoder...");
        let latent_decoder: LatentDecoder<burn_wgpu::Wgpu> =
            load_latent_decoder_model(&opts.model_dir, &device).unwrap();

        info!("Running encoder...");

        let latent = latent_decoder.image_to_latent(&imgs, &device);

        // get converted pixels idxs
        let [_, _, height, width] = latent.dims();
        let scale = imgs.height / height;
        let crop_left = crop_left / scale;
        let crop_right = crop_right / scale;
        let crop_top = crop_top / scale;
        let crop_bottom = crop_bottom / scale;

        // compute mask
        let crop_width = crop_right - crop_left;
        let crop_height = crop_bottom - crop_top;

        let pad_left = crop_left;
        let pad_right = width - crop_right;

        let pad_top = crop_top;
        let pad_bottom = height - crop_bottom;

        let mask = Tensor::<_, 2>::ones([crop_height, crop_width], &device)
            .pad((pad_left, pad_right, pad_top, pad_bottom), 0.0)
            .bool()
            .unsqueeze::<4>()
            .expand([1, 4, height, width]);
        let mask = if opts.crop_out { mask.bool_not() } else { mask };

        InpaintingTensors {
            orig_dims: (imgs.width, imgs.height),
            reference_latent: latent,
            mask: mask.unsqueeze::<4>(),
        }
    });

    /*let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        einfo!("Usage: {} <model_dir> <refiner(y/n)> <unconditional_guidance_scale> <n_diffusion_steps> <prompt> <output_image_name>", args[0]);
        process::exit(1);
    }*/

    /*let unconditional_guidance_scale: f64 = args[3].parse().unwrap_or_else(|_| {
        einfo!("Error: Invalid unconditional guidance scale.");
        process::exit(1);
    });
    let n_steps: usize = args[4].parse().unwrap_or_else(|_| {
        einfo!("Error: Invalid number of diffusion steps.");
        process::exit(1);
    });
    let prompt = &args[5];
    let output_image_name = &args[6];*/

    let conditioning = {
        info!("Loading embedder...");
        let embedder: Embedder<burn_wgpu::Wgpu> =
            load_embedder_model(&opts.model_dir, &device).unwrap();

        let resolution = if let Some(inpainting_info) = inpainting_info.as_ref() {
            [
                inpainting_info.orig_dims.1 as i32,
                inpainting_info.orig_dims.0 as i32,
            ]
        } else {
            [1024, 1024]
        }; //RESOLUTIONS[8];

        let size = Tensor::from_ints(resolution, &device).unsqueeze();
        let crop = Tensor::from_ints([0, 0], &device).unsqueeze();
        let ar = Tensor::from_ints(resolution, &device).unsqueeze();

        info!("Running embedder...");
        embedder.text_to_conditioning(&opts.prompt, size, crop, ar)
    };

    let conditioning: Conditioning<_> = conditioning.with_backend(&device);

    let latent = {
        info!("Loading diffuser...");
        let diffuser: Diffuser<_> = load_diffuser_model(&opts.model_dir, &device).unwrap();

        if let Some(inpainting_info) = inpainting_info {
            diffuser.sample_latent_with_inpainting(
                conditioning.clone(),
                opts.unconditional_guidance_scale,
                opts.n_diffusion_steps,
                inpainting_info.reference_latent,
                inpainting_info.mask,
            )
        } else {
            info!("Running diffuser...");
            diffuser.sample_latent(
                conditioning.clone(),
                opts.unconditional_guidance_scale,
                opts.n_diffusion_steps,
            )
        }
    };

    let latent = if opts.use_refiner {
        info!("Loading refiner...");
        let diffuser: Diffuser<_> = load_diffuser_model(&opts.model_dir, &device).unwrap();

        info!("Running refiner...");
        diffuser.refine_latent(
            latent,
            conditioning,
            opts.unconditional_guidance_scale,
            800,
            opts.n_diffusion_steps,
        )
    } else {
        latent
    };

    let images = {
        info!("Loading latent decoder...");
        let latent_decoder: LatentDecoder<_> =
            load_latent_decoder_model(&opts.model_dir, &device).unwrap();

        info!("Running decoder...");
        latent_decoder.latent_to_image(latent)
    };

    info!("Saving images...");
    save_images(
        &images.buffer,
        opts.output_dir.to_str().unwrap(),
        images.width as u32,
        images.height as u32,
    )
    .unwrap();
    info!("Done.");

    return;
}

use image::io::Reader as ImageReader;
use image::{self, ColorType::Rgb8, ImageError, ImageResult, RgbImage};

fn load_images(filenames: &[String]) -> Result<RawImages, ImgLoadError> {
    let images = filenames
        .into_iter()
        .map(|filename| load_image(&filename))
        .collect::<ImageResult<Vec<RgbImage>>>()?;

    let (width, height) = images
        .first()
        .map(|img| img.dimensions())
        .ok_or(ImgLoadError::NoImages)?;

    if !images
        .iter()
        .map(|img| img.dimensions())
        .all(|d| d == (width, height))
    {
        return Err(ImgLoadError::DifferentDimensions);
    }

    let image_buffers: Vec<Vec<u8>> = images.into_iter().map(|image| image.into_vec()).collect();

    Ok(RawImages {
        buffer: image_buffers,
        width: width as usize,
        height: height as usize,
    })
}

#[derive(Debug)]
enum ImgLoadError {
    DifferentDimensions,
    NoImages,
    ImageError(ImageError),
}

impl From<ImageError> for ImgLoadError {
    fn from(err: ImageError) -> Self {
        ImgLoadError::ImageError(err)
    }
}

fn load_image(filename: &str) -> ImageResult<RgbImage> {
    Ok(ImageReader::open(filename)?.decode()?.to_rgb8())
}

fn save_images(images: &Vec<Vec<u8>>, basepath: &str, width: u32, height: u32) -> ImageResult<()> {
    for (index, img_data) in images.iter().enumerate() {
        let path = format!("{}{}.png", basepath, index);
        image::save_buffer(path, &img_data[..], width, height, Rgb8)?;
    }

    Ok(())
}

// save red test image
fn save_test_image() -> ImageResult<()> {
    let width = 256;
    let height = 256;
    let raw: Vec<_> = (0..width * height)
        .into_iter()
        .flat_map(|i| {
            let row = i / width;
            let red = (255.0 * row as f64 / height as f64) as u8;

            [red, 0, 0]
        })
        .collect();

    image::save_buffer("red.png", &raw[..], width, height, Rgb8)
}

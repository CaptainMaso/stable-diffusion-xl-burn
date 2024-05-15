use crate::prelude::*;

use burn::module::Param;
use burn_ndarray::{NdArray, NdArrayDevice::Cpu as NdCpu};
use ndarray_npy as npy;
use num_traits::cast::ToPrimitive;
use std::path::Path;

// pub fn numpy_to_tensor<B: Backend, const D: usize>(
//     numpy_data: NpyData<f32>,
//     device: &B::Device,
// ) -> Tensor<B, D> {
//     let v = numpy_data.to_vec();

//     let shape: Vec<_> = v[0..D].into_iter().map(|&v| v as usize).collect();
//     let data: Vec<B::FloatElem> = v[D..].into_iter().map(|e| e.elem()).collect();

//     Tensor::from_data(Data::new(data, shape.into()), device)
// }

pub fn load_tensor<E, const D: usize>(name: &str, path: &Path) -> Result<Tensor<NdArray, D>>
where
    E: npy::ReadableElement + Clone + Default,
{
    let np_tensor: ndarray::ArrayD<_> =
        npy::read_npy(path).with_context(|| anyhow!("Reading {path:?}"))?;
    let shape: [usize; D] = np_tensor.shape().try_into().unwrap();
    let data = Data::new(np_tensor.into_raw_vec(), shape.into());
    let tensor = Tensor::from_data(data, &NdCpu);
    Ok(tensor)
}

pub fn load_scalar(name: &str, path: &Path) -> Result<f32> {
    load_tensor::<f32, 1>(name, path).and_then(|t| {
        if t.shape().num_elements() == 1 {
            Ok(t.into_scalar())
        } else {
            Err(anyhow!(
                "Tensor '{name}' @ {path:?} has more than 1 element. Cannot load as a scalar"
            ))
        }
    })
}

pub fn load_i64(name: &str, path: &Path) -> Result<i64> {
    Ok(load_scalar(name, path)? as i64)
}

pub fn load_linear(path: &Path) -> Result<nn::Linear<NdArray>> {
    let weight = load_tensor::<f32, 2>("weight", path)?;
    let bias = load_tensor::<f32, 1>("bias", path).ok();

    let mut linear: nn::Linear<_> = nn::LinearConfig::new(3, 3).init(&NdCpu);
    linear.weight = Param::from_tensor(weight);
    linear.bias = bias.map(Param::from_tensor);

    Ok(linear)
}

pub fn load_embedding(path: &Path) -> Result<nn::Embedding<NdArray>> {
    let weight = load_tensor::<f32, 2>("weight", path)?;
    let [n_vocab, n_state] = weight.dims();

    let mut embedding = nn::EmbeddingConfig::new(n_vocab, n_state).init(&NdCpu);
    embedding.weight = Param::from_tensor(weight);

    Ok(embedding)
}

/*pub fn load_layer_norm<B: Backend>(path: &str, device: &B::Device) -> Result<nn::LayerNorm<B>> {
    let weight = load_tensor::<B, 1>("weight", path, device)?;
    let bias = load_tensor::<B, 1>("bias", path, device)?;
    let eps = load_f32::<B>("eps", path, device)? as f64;

    let [n_state] = weight.dims();

    let record = nn::LayerNormRecord {
        gamma: weight.into(),
        beta: bias.into(),
        epsilon: <f64 as Module<B>>::into_record(eps),
    };

    let layer_norm: nn::LayerNorm<B> = nn::LayerNormConfig::new(n_state).init_with(record);

    Ok(layer_norm)
}*/

/*pub fn load_rmsnorm<B: Backend>(path: &str, device: &B::Device) -> Result<RMSNorm<B>> {
    let weight = load_tensor::<B, 1>("weight", path, device)?;
    let eps = load_f32::<B>("eps", path, device)?.into();

    let rmsnorm =  RMSNorm {
        weight: weight.into(),
        eps: eps
    };

    Ok(rmsnorm)
}*/

pub fn load_conv2d<B: Backend>(path: &str, device: &B::Device) -> Result<conv::Conv2d<B>> {
    let weight = load_tensor::<B, 4>("weight", path, device)?;
    let bias = load_tensor::<B, 1>("bias", path, device).ok();
    let has_bias = bias.is_some();

    let stride = load_tensor::<B, 1>("stride", path, device)?;
    let stride = tensor_to_array_2(stride);

    let kernel_size = load_tensor::<B, 1>("kernel_size", path, device)?;
    let kernel_size = tensor_to_array_2(kernel_size);

    let dilation = load_tensor::<B, 1>("dilation", path, device)?;
    let dilation = tensor_to_array_2(dilation);

    let n_group = load_i64::<B>("n_group", path, device)?.into();
    let n_channels_in = load_i64::<B>("n_channels_in", path, device)?.into();
    let n_channels_out = load_i64::<B>("n_channels_out", path, device)?.into();

    let padding = load_tensor::<B, 1>("padding", path, device)?;
    let padding = tensor_to_array_2(padding);
    let padding = nn::PaddingConfig2d::Explicit(padding[0], padding[1]);

    let mut conv2d: conv::Conv2d<B> =
        conv::Conv2dConfig::new([n_channels_in, n_channels_out], kernel_size)
            .with_stride(stride)
            .with_dilation(dilation)
            .with_groups(n_group)
            .with_padding(padding)
            .with_bias(has_bias)
            .init(device);
    conv2d.weight = Param::from_tensor(weight);
    conv2d.bias = bias.map(Param::from_tensor);

    Ok(conv2d)
}

pub fn tensor_to_array_2<B: Backend>(x: Tensor<B, 1>) -> [usize; 2] {
    let vec = x.into_data().value;
    assert!(vec.len() == 2, "Tensor length must be 2.");
    [vec[0].to_usize().unwrap(), vec[1].to_usize().unwrap()]
}

pub fn tensor_to_array<const N: usize, B: Backend>(x: Tensor<B, 1>) -> [usize; N] {
    let vec = x.into_data().value;
    assert!(vec.len() == N, "Tensor length must be {}.", N);

    let mut arr = [0; N];
    for (a, t) in arr.iter_mut().zip(vec) {
        *a = t.to_usize().unwrap();
    }

    arr
}

// #[cfg(feature = "load")]
// pub mod load;

use crate::prelude::*;

use burn::{
    config::Config,
    module::{Module, Param},
    tensor::{backend::Backend, Tensor},
};

#[derive(Config)]
pub struct LayerNormConfig {
    d_size: usize,
    #[config(default = 1e-5)]
    eps: f64,
}

impl LayerNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LayerNorm<B> {
        let gamma = Param::from_tensor(Tensor::ones([self.d_size], device));
        let beta = Param::from_tensor(Tensor::zeros([self.d_size], device));

        let eps = self.eps;

        LayerNorm { gamma, beta, eps }
    }
}

#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    gamma: Param<Tensor<B, 1>>,
    beta: Param<Tensor<B, 1>>,
    eps: f64,
}

impl<B: Backend> LayerNorm<B> {
    // #[cfg(feature = "load")]
    // pub fn load(path: &Path, name: &str, device: &B::Device) -> Result<Self> {
    //     let args = burn_import::pytorch::LoadArgs::new(path.to_path_buf()).with_top_level_key(name);
    //     let record = burn_import::pytorch::PyTorchFileRecorder::new().load(args, device)?;
    //     Ok(Self::load_record(Self::default(), record))
    // }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        layernorm(x, self.eps)
            .mul(self.gamma.val().unsqueeze())
            .add(self.beta.val().unsqueeze())
    }
}

pub fn layernorm<B: Backend, const D: usize>(x: Tensor<B, D>, eps: f64) -> Tensor<B, D> {
    //let (var, mean) = x.clone().var_mean_bias(D - 1);
    //x.sub(mean).div(var.sqrt().add_scalar(eps))

    let u = x.clone() - x.mean_dim(D - 1);
    u.clone()
        .div((u.clone() * u).mean_dim(D - 1).add_scalar(eps).sqrt())
}

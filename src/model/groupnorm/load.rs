use std::path::Path;

use super::*;
use crate::prelude::*;

use burn_import::pytorch::{config_from_file, LoadArgs, PyTorchFileRecorder};

pub fn load_group_norm<B: Backend>(
    path: &Path,
    name: &str,
    device: &B::Device,
) -> Result<GroupNorm<B>> {
    

    GroupNorm::load_file(self, file_path, recorder, device)

    // let n_group = config_from_file::<B>(path, Some("n_group"), device)?.into();
    // let n_channel = config_from_file::<B>(path, Some("n_channel"), device)?.into();
    // let eps = config_from_file::<B>(path, Some("eps"), device)?.into();

    // let gamma = Param::from_tensor(
    //     config_from_file::<B>(path, Some("weight"), device)
    //         .ok()
    //         .unwrap_or_else(|| Tensor::ones([n_channel], device)),
    // );
    // let beta = Param::from_tensor(
    //     config_from_file::<B>(path, Some("bias"), device)
    //         .ok()
    //         .unwrap_or_else(|| Tensor::zeros([n_channel], device)),
    // );

    Ok(GroupNorm {
        n_group,
        n_channel,
        gamma,
        beta,
        eps,
    })
}

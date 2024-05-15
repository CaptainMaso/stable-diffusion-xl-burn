pub mod backend;
pub mod model;
pub mod token;
pub mod with_backend;

pub mod prelude {
    pub(crate) use std::path::Path;

    pub(crate) use burn::prelude::*;

    pub(crate) use anyhow::{anyhow, Context, Result};
}

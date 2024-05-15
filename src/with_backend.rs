use burn::{prelude::*, tensor::Element};

pub trait WithBackend<B: Backend> {
    type Output;

    fn with_backend(&self, device: &B::Device) -> Self::Output;
}

impl<B, B2, K, const D: usize> WithBackend<B2> for Tensor<B, D, K>
where
    B: Backend,
    B2: Backend,
    K: burn::tensor::TensorKind<B>
        + burn::tensor::BasicOps<B>
        + burn::tensor::TensorKind<B2>
        + burn::tensor::BasicOps<B2>,
    <K as burn::tensor::BasicOps<B>>::Elem: Element,
    <K as burn::tensor::BasicOps<B2>>::Elem: Element,
{
    type Output = Tensor<B2, D, K>;

    #[inline]
    fn with_backend(&self, device: &B2::Device) -> Self::Output {
        Tensor::<B2, D, K>::from_data(self.to_data().convert(), device)
    }
}

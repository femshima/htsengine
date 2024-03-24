use super::{
    coefficients::{Coefficients, GeneralizedCoefficients},
    mglsa::MelGeneralizedLogSpectrumApproximation,
    mlsa::MelLogSpectrumApproximation,
};

#[derive(Debug, Clone)]
pub enum Stage {
    NonZero {
        stage: usize,
        gamma: f64,
        coefficients: GeneralizedCoefficients,
        filter: MelGeneralizedLogSpectrumApproximation,
    },
    Zero {
        coefficients: Coefficients,
        filter: MelLogSpectrumApproximation<6>,
    },
}

impl Stage {
    pub fn new(stage: usize, c_len: usize) -> Self {
        if stage == 0 {
            Self::Zero {
                coefficients: Coefficients { buffer: Vec::new() },
                filter: MelLogSpectrumApproximation::new(c_len),
            }
        } else {
            let gamma = -1.0 / stage as f64;
            Self::NonZero {
                stage,
                gamma,
                coefficients: GeneralizedCoefficients {
                    buffer: Vec::new(),
                    gamma,
                },
                filter: MelGeneralizedLogSpectrumApproximation::new(stage, c_len),
            }
        }
    }
}

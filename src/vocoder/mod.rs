use crate::constants::{MAX_F0, MAX_LF0, MIN_F0, MIN_LF0, NODATA};

#[macro_use]
mod buffer;
mod cepstrum;
mod coefficients;
mod excitation;
mod generalized;
mod lsp;
mod mglsa;
mod mlsa;
mod stage;

use self::{
    buffer::Buffer,
    cepstrum::{CepstrumT, MelCepstrum},
    excitation::Excitation,
    generalized::Generalized,
    lsp::LineSpectralPairs,
    stage::Stage,
};

#[derive(Debug, Clone)]
pub struct Vocoder {
    stage: Stage,

    use_log_gain: bool,
    fperiod: usize,
    rate: usize,

    alpha: f64,
    beta: f64,
    volume: f64,

    excitation: Option<Excitation>,
}

impl Vocoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        m: usize,
        stage: usize,
        use_log_gain: bool,
        rate: usize,
        fperiod: usize,
        alpha: f64,
        beta: f64,
        volume: f64,
    ) -> Self {
        let stage = Stage::new(stage, m + 1);

        Self {
            stage,

            use_log_gain,
            fperiod,
            rate,

            alpha,
            beta,
            volume,

            excitation: None,
        }
    }

    pub fn synthesize(&mut self, lf0: f64, spectrum: &[f64], lpf: &[f64], rawdata: &mut [f64]) {
        let p = if lf0 == NODATA {
            0.0
        } else if lf0 <= MIN_LF0 {
            self.rate as f64 / MIN_F0
        } else if lf0 >= MAX_LF0 {
            self.rate as f64 / MAX_F0
        } else {
            self.rate as f64 / lf0.exp()
        };
        if self.excitation.is_none() {
            match self.stage {
                Stage::Zero {
                    ref mut coefficients,
                    ..
                } => {
                    let cepstrum = MelCepstrum::new(spectrum, self.alpha);
                    *coefficients = cepstrum.mc2b();
                }
                Stage::NonZero {
                    stage,
                    gamma,
                    ref mut coefficients,
                    ..
                } => {
                    let lsp = LineSpectralPairs::new(
                        spectrum,
                        self.alpha,
                        self.use_log_gain,
                        stage,
                        gamma,
                    );
                    *coefficients = lsp.lsp2mgc().mc2b().gnorm();
                    for i in 1..coefficients.len() {
                        coefficients[i] *= gamma;
                    }
                }
            }
        }

        match self.stage {
            Stage::Zero {
                ref mut coefficients,
                ref mut filter,
            } => {
                let mut cepstrum = MelCepstrum::new(spectrum, self.alpha);
                cepstrum.postfilter_mcp(self.beta);
                let cc = cepstrum.mc2b();
                let cinc: Vec<_> = cc
                    .iter()
                    .zip(&*coefficients)
                    .map(|(cc, c)| (cc - c) / self.fperiod as f64)
                    .collect();

                let excitation = self
                    .excitation
                    .get_or_insert_with(|| Excitation::new(p, lpf.len()));
                excitation.start(p, self.fperiod);

                (0..self.fperiod).for_each(|i| {
                    let mut x = excitation.get(lpf);
                    if x != 0.0 {
                        x *= coefficients[0].exp();
                    }
                    filter.df(&mut x, self.alpha, coefficients);
                    for i in 0..coefficients.len() {
                        coefficients[i] += cinc[i];
                    }
                    rawdata[i] = x * self.volume;
                });

                excitation.end(p);
                *coefficients = cc;
            }
            Stage::NonZero {
                stage,
                gamma,
                ref mut coefficients,
                ref mut filter,
            } => {
                let mut lsp =
                    LineSpectralPairs::new(spectrum, self.alpha, self.use_log_gain, stage, gamma);
                lsp.postfilter_lsp(self.beta);
                lsp.check_lsp_stability();
                let mut cc = lsp.lsp2mgc().mc2b().gnorm();
                for i in 1..cc.len() {
                    cc[i] *= gamma;
                }
                let cinc: Vec<_> = cc
                    .iter()
                    .zip(&*coefficients)
                    .map(|(cc, c)| (cc - c) / self.fperiod as f64)
                    .collect();

                let excitation = self
                    .excitation
                    .get_or_insert_with(|| Excitation::new(p, lpf.len()));
                excitation.start(p, self.fperiod);

                (0..self.fperiod).for_each(|i| {
                    let mut x = excitation.get(lpf);
                    x *= coefficients[0];
                    filter.df(&mut x, self.alpha, coefficients);
                    for i in 0..coefficients.len() {
                        coefficients[i] += cinc[i];
                    }
                    rawdata[i] = x * self.volume;
                });

                excitation.end(p);
                *coefficients = cc;
            }
        }
    }
}

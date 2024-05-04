use crate::vocoder::Vocoder;

type Parameter = Vec<Vec<f64>>;

pub struct SpeechGenerator {
    fperiod: usize,
    alpha: f64,
    beta: f64,
    volume: f64,
}

impl SpeechGenerator {
    pub fn new(fperiod: usize, alpha: f64, beta: f64, volume: f64) -> Self {
        Self {
            fperiod,
            alpha,
            beta,
            volume,
        }
    }
    /// Generate speech
    pub fn synthesize(
        &self,
        mut v: Vocoder,
        spectrum: Parameter,
        lf0: Parameter,
        lpf: Option<Parameter>,
    ) -> Vec<f64> {
        // check
        if lf0.len() > 0 {
            if lf0[0].len() != 1 {
                panic!("The size of lf0 static vector must be 1.");
            }
            if lpf.as_ref().map(|lpf| lpf[0].len() % 2 == 0) == Some(true) {
                panic!("The number of low-pass filter coefficient must be odd numbers.");
            }
        }

        // create speech buffer
        let total_frame = lf0.len();
        let mut speech = vec![0.0; total_frame * self.fperiod];

        // synthesize speech waveform
        for i in 0..total_frame {
            v.synthesize(
                lf0[i][0],
                &spectrum[i],
                lpf.as_ref().map(|lpf| &lpf[i] as &[f64]).unwrap_or(&[]),
                self.alpha,
                self.beta,
                self.volume,
                &mut speech[i * self.fperiod..(i + 1) * self.fperiod],
            );
        }

        speech
    }
}

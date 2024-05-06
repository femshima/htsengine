use std::path::Path;
use std::sync::Arc;

use crate::constants::DB;
use crate::duration::DurationEstimator;
use crate::label::{LabelError, Labels};
use crate::mlpg_adjust::MlpgAdjust;
use crate::model::interporation_weight::InterporationWeight;
use crate::model::{apply_additional_half_tone, ModelError, Models, VoiceSet};
use crate::speech::SpeechGenerator;
use crate::vocoder::Vocoder;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("Model error: {0}")]
    ModelError(#[from] ModelError),
    #[error("Failed to parse option {0}")]
    ParseOptionError(String),

    #[error("Label error: {0}")]
    LabelError(#[from] LabelError),
}

#[derive(Debug, Clone)]
pub struct Condition {
    /// Sampling frequency
    sampling_frequency: usize,
    /// Frame period
    fperiod: usize,
    /// Volume
    volume: f64,
    /// MSD thresholds
    msd_threshold: Vec<f64>,
    /// GV weights
    gv_weight: Vec<f64>,
    /// Flag for using phoneme alignment in label
    phoneme_alignment_flag: bool,
    /// Speech speed
    speed: f64,
    /// If stage = 0 then gamma = 0 else gamma = -1/stage
    stage: usize,
    /// Log gain flag (for LSP)
    use_log_gain: bool,
    /// All-pass constant
    alpha: f64,
    /// Postfiltering coefficient
    beta: f64,
    /// Additional half tone
    additional_half_tone: f64,

    /// Interporation weights
    interporation_weight: InterporationWeight,
}

impl Default for Condition {
    fn default() -> Self {
        Self {
            sampling_frequency: 0,
            fperiod: 0,
            volume: 1.0f64,
            msd_threshold: Vec::new(),
            gv_weight: Vec::new(),
            speed: 1.0f64,
            phoneme_alignment_flag: false,
            stage: 0,
            use_log_gain: false,
            alpha: 0.0f64,
            beta: 0.0f64,
            additional_half_tone: 0.0f64,
            interporation_weight: InterporationWeight::default(),
        }
    }
}

impl Condition {
    pub fn load_model(&mut self, voices: &VoiceSet) -> Result<(), EngineError> {
        let metadata = voices.global_metadata();

        let nstream = metadata.num_streams;

        /* global */
        self.sampling_frequency = metadata.sampling_frequency;
        self.fperiod = metadata.frame_period;
        self.msd_threshold = [0.5].repeat(nstream);
        self.gv_weight = [1.0].repeat(nstream);

        /* spectrum */
        for option in &voices.stream_metadata(0).option {
            let Some((key, value)) = option.split_once('=') else {
                eprintln!("Skipped unrecognized option {}.", option);
                continue;
            };
            match key {
                "GAMMA" => {
                    self.stage = value
                        .parse()
                        .map_err(|_| EngineError::ParseOptionError(key.to_string()))?
                }
                "LN_GAIN" => match value {
                    "1" => self.use_log_gain = true,
                    "0" => self.use_log_gain = false,
                    _ => return Err(EngineError::ParseOptionError(key.to_string())),
                },
                "ALPHA" => {
                    self.alpha = value
                        .parse()
                        .map_err(|_| EngineError::ParseOptionError(key.to_string()))?
                }
                _ => eprintln!("Skipped unrecognized option {}.", option),
            }
        }

        /* interpolation weights */
        self.interporation_weight = InterporationWeight::new(voices.len(), nstream);

        Ok(())
    }

    /// Set sampling frequency (Hz), 1 <= i
    pub fn set_sampling_frequency(&mut self, i: usize) {
        self.sampling_frequency = i.max(1);
    }
    /// Get sampling frequency
    pub fn get_sampling_frequency(&self) -> usize {
        self.sampling_frequency
    }

    /// Set frame shift (point), 1 <= i
    pub fn set_fperiod(&mut self, i: usize) {
        self.fperiod = i.max(1);
    }
    /// Get frame shift (point)
    pub fn get_fperiod(&self) -> usize {
        self.fperiod
    }

    /// Set volume in db
    /// Note: Default value is 0.0.
    pub fn set_volume(&mut self, f: f64) {
        self.volume = (f * DB).exp();
    }
    /// Get volume in db
    pub fn get_volume(&self) -> f64 {
        self.volume.ln() / DB
    }

    /// Set threshold for MSD
    /// Note: Default value is 0.5.
    pub fn set_msd_threshold(&mut self, stream_index: usize, f: f64) {
        self.msd_threshold[stream_index] = f.clamp(0.0, 1.0);
    }
    /// Get threshold for MSD
    pub fn get_msd_threshold(&self, stream_index: usize) -> f64 {
        self.msd_threshold[stream_index]
    }

    /// Set GV weight
    /// Note: Default value is 1.0.
    pub fn set_gv_weight(&mut self, stream_index: usize, f: f64) {
        self.gv_weight[stream_index] = f.max(0.0);
    }
    /// Get GV weight
    pub fn get_gv_weight(&self, stream_index: usize) -> f64 {
        self.gv_weight[stream_index]
    }

    /// Set speed
    /// Note: Default value is 1.0.
    pub fn set_speed(&mut self, f: f64) {
        self.speed = f.max(1.0E-06);
    }
    /// Get speed
    pub fn get_speed(&self) -> f64 {
        self.speed
    }

    /// Set flag to use phoneme alignment in label
    /// Note: Default value is 1.0.
    pub fn set_phoneme_alignment_flag(&mut self, b: bool) {
        self.phoneme_alignment_flag = b;
    }
    /// Get flag to use phoneme alignment in label
    pub fn get_phoneme_alignment_flag(&self) -> bool {
        self.phoneme_alignment_flag
    }

    /// Set frequency warping parameter alpha
    pub fn set_alpha(&mut self, f: f64) {
        self.alpha = f.clamp(0.0, 1.0);
    }
    /// Get frequency warping parameter alpha
    pub fn get_alpha(&self) -> f64 {
        self.alpha
    }

    /// Set postfiltering coefficient parameter beta
    pub fn set_beta(&mut self, f: f64) {
        self.beta = f.clamp(0.0, 1.0);
    }
    /// Get postfiltering coefficient parameter beta
    pub fn get_beta(&self) -> f64 {
        self.beta
    }

    /// Set additional half tone
    pub fn set_additional_half_tone(&mut self, f: f64) {
        self.additional_half_tone = f;
    }
    /// Get additional half tone
    pub fn get_additional_half_tone(&self) -> f64 {
        self.additional_half_tone
    }

    /// Get interporation weight
    pub fn get_interporation_weight(&self) -> &InterporationWeight {
        &self.interporation_weight
    }
    /// Get interporation weight as mutable reference
    pub fn get_interporation_weight_mut(&mut self) -> &mut InterporationWeight {
        &mut self.interporation_weight
    }
}

pub struct Engine {
    pub condition: Condition,
    pub voices: VoiceSet,
}

impl Engine {
    #[cfg(feature = "htsvoice")]
    pub fn load<P: AsRef<Path>>(voices: &[P]) -> Result<Self, EngineError> {
        use crate::model::load_htsvoice_file;

        let voices = voices
            .iter()
            .map(|path| Ok(Arc::new(load_htsvoice_file(path)?)))
            .collect::<Result<Vec<_>, ModelError>>()?;
        let voiceset = VoiceSet::new(voices)?;

        let mut condition = Condition::default();
        condition.load_model(&voiceset)?;

        Ok(Self::new(voiceset, condition))
    }
    pub fn new(voices: VoiceSet, condition: Condition) -> Self {
        Engine { voices, condition }
    }

    pub fn synthesize_from_strings<S: AsRef<str>>(
        &self,
        lines: &[S],
    ) -> Result<Vec<f64>, EngineError> {
        let labels = Labels::load_from_strings(
            self.condition.sampling_frequency,
            self.condition.fperiod,
            lines,
        )?;
        Ok(self.generate_speech(&labels))
    }

    pub fn synthesize_from_labels(
        &self,
        labels: Vec<jlabel::Label>,
    ) -> Result<Vec<f64>, EngineError> {
        let labels = Labels::new(labels, None)?;
        Ok(self.generate_speech(&labels))
    }

    pub fn generate_speech(&self, labels: &Labels) -> Vec<f64> {
        let vocoder = Vocoder::new(
            self.voices.stream_metadata(0).vector_length,
            self.voices.stream_metadata(2).vector_length,
            self.condition.stage,
            self.condition.use_log_gain,
            self.condition.sampling_frequency,
            self.condition.alpha,
            self.condition.beta,
            self.condition.volume,
            self.condition.fperiod,
        );

        let models = Models::new(
            labels.labels(),
            &self.voices,
            &self.condition.interporation_weight,
        );

        let estimator = DurationEstimator::new(models.duration(), models.nstate());
        let durations = if self.condition.phoneme_alignment_flag {
            estimator.create_with_alignment(labels.times())
        } else {
            estimator.create(self.condition.speed)
        };

        let spectrum =
            MlpgAdjust::new(self.condition.gv_weight[0], self.condition.msd_threshold[0]).create(
                models.stream(0),
                models.vector_length(0),
                models.windows(0),
                models.gv(0),
                &durations,
            );
        let lf0 = MlpgAdjust::new(self.condition.gv_weight[1], self.condition.msd_threshold[1])
            .create(
                mutated(models.stream(1), |params| {
                    apply_additional_half_tone(params, self.condition.additional_half_tone);
                }),
                models.vector_length(1),
                models.windows(1),
                models.gv(1),
                &durations,
            );
        let lpf = MlpgAdjust::new(self.condition.gv_weight[2], self.condition.msd_threshold[2])
            .create(
                models.stream(2),
                models.vector_length(2),
                models.windows(2),
                models.gv(2),
                &durations,
            );

        let generator = SpeechGenerator::new(self.condition.fperiod);
        generator.synthesize(vocoder, spectrum, lf0, lpf)
    }
}

fn mutated<T, F: FnOnce(&mut T)>(mut value: T, f: F) -> T {
    f(&mut value);
    value
}

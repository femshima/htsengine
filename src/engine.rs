use std::path::Path;
use std::sync::Arc;

use crate::constants::{DB, HALF_TONE, MAX_LF0, MIN_LF0};
use crate::gstream::GenerateSpeechStreamSet;
use crate::label::Label;
use crate::model::interporation_weight::InterporationWeight;
use crate::model::{ModelError, ModelSet};
use crate::pstream::ParameterStreamSet;
use crate::sstream::StateStreamSet;
use crate::vocoder::Vocoder;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("Model error")]
    ModelError(#[from] ModelError),
    #[error("Failed to parse option {0}")]
    ParseOptionError(String),
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
    pub fn load_model(&mut self, ms: &ModelSet) -> Result<(), EngineError> {
        let nvoices = ms.get_nvoices();
        let nstream = ms.get_nstream();

        /* global */
        self.sampling_frequency = ms.get_sampling_frequency();
        self.fperiod = ms.get_fperiod();
        self.msd_threshold = [0.5].repeat(nstream);
        self.gv_weight = [1.0].repeat(nstream);

        /* spectrum */
        for option in ms.get_option(0).unwrap_or(&[]) {
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
                "LN_GAIN" => self.use_log_gain = value == "1",
                "ALPHA" => {
                    self.alpha = value
                        .parse()
                        .map_err(|_| EngineError::ParseOptionError(key.to_string()))?
                }
                _ => eprintln!("Skipped unrecognized option {}.", option),
            }
        }

        /* interpolation weights */
        self.interporation_weight = InterporationWeight::new(nvoices, nstream);

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
        self.msd_threshold[stream_index] = f.min(1.0).max(0.0);
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
        self.alpha = f.max(0.0).min(1.0);
    }
    /// Get frequency warping parameter alpha
    pub fn get_alpha(&self) -> f64 {
        self.alpha
    }

    /// Set postfiltering coefficient parameter beta
    pub fn set_beta(&mut self, f: f64) {
        self.beta = f.max(0.0).min(1.0);
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
    pub ms: Arc<ModelSet>,
}

impl Engine {
    pub fn load<P: AsRef<Path>>(voices: &[P]) -> Result<Engine, EngineError> {
        let ms = ModelSet::load_htsvoice_files(voices).unwrap();
        let mut condition = Condition::default();
        condition.load_model(&ms)?;
        Ok(Self::new(Arc::new(ms), condition))
    }
    pub fn new(ms: Arc<ModelSet>, condition: Condition) -> Engine {
        Engine { condition, ms }
    }

    pub fn synthesize_from_strings(&self, lines: &[String]) -> GenerateSpeechStreamSet {
        let labels = self.load_labels(lines);
        let state_sequence = self.generate_state_sequence(&labels);
        let parameter_sequence = self.generate_parameter_sequence(&state_sequence);
        self.generate_sample_sequence(&parameter_sequence)
    }

    fn load_labels(&self, lines: &[String]) -> Label {
        Label::load_from_strings(
            self.condition.sampling_frequency,
            self.condition.fperiod,
            lines,
        )
    }

    fn generate_state_sequence(&self, label: &Label) -> StateStreamSet {
        let mut sss = StateStreamSet::create(
            self.ms.clone(),
            label,
            self.condition.phoneme_alignment_flag,
            self.condition.speed,
            &self.condition.interporation_weight,
        );
        self.apply_additional_half_tone(&mut sss);
        sss
    }

    fn apply_additional_half_tone(&self, sss: &mut StateStreamSet) {
        if self.condition.additional_half_tone == 0.0 {
            return;
        }
        for i in 0..sss.get_total_state() {
            let mut f = sss.get_mean(1, i, 0);
            f += self.condition.additional_half_tone * HALF_TONE;
            f = f.max(MIN_LF0).min(MAX_LF0);
            sss.set_mean(1, i, 0, f);
        }
    }

    fn generate_parameter_sequence(&self, state_sequence: &StateStreamSet) -> ParameterStreamSet {
        ParameterStreamSet::create(
            state_sequence,
            &self.condition.msd_threshold,
            &self.condition.gv_weight,
        )
    }

    fn generate_sample_sequence(
        &self,
        parameter_sequence: &ParameterStreamSet,
    ) -> GenerateSpeechStreamSet {
        let vocoder = Vocoder::new(
            self.ms.get_vector_length(0) - 1,
            self.condition.stage,
            self.condition.use_log_gain,
            self.condition.sampling_frequency,
            self.condition.fperiod,
        );
        GenerateSpeechStreamSet::create(
            parameter_sequence,
            vocoder,
            self.condition.fperiod,
            self.condition.alpha,
            self.condition.beta,
            self.condition.volume,
        )
    }
}

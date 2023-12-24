use crate::{pstream::PStreamSet, vocoder::Vocoder};

pub struct GStreamSet {
    speech: Vec<f64>,
}

impl GStreamSet {
    /// create: generate speech
    pub fn create(
        pss: &PStreamSet,
        stage: usize,
        use_log_gain: bool,
        sampling_rate: usize,
        fperiod: usize,
        alpha: f64,
        beta: f64,
        volume: f64,
    ) -> Self {
        // check
        if pss.get_nstream() != 2 && pss.get_nstream() != 3 {
            panic!("The number of streams must be 2 or 3.");
        }
        if pss.get_vector_length(1) != 1 {
            panic!("The size of lf0 static vector must be 1.");
        }
        if pss.get_nstream() >= 3 && pss.get_vector_length(2) % 2 == 0 {
            panic!("The number of low-pass filter coefficient must be odd numbers.");
        }

        // create speech buffer
        let total_frame = pss.get_total_frame();
        let mut speech = vec![0.0; total_frame * fperiod];

        // synthesize speech waveform
        let mut v = Vocoder::new(
            pss.get_vector_length(0) - 1,
            stage as usize,
            use_log_gain,
            sampling_rate as usize,
            fperiod as usize,
        );
        let nlpf = if pss.get_nstream() >= 3 {
            pss.get_vector_length(2)
        } else {
            0
        };

        let mut frame_skipped_index = vec![0; pss.get_nstream()];
        for i in 0..total_frame {
            let get_parameter = |stream_index: usize, vector_index: usize| {
                if pss.is_msd(stream_index) && !pss.get_msd_flag(stream_index, i) {
                    -1e10 // HTS_NODATA
                } else {
                    pss.get_parameter(
                        stream_index,
                        frame_skipped_index[stream_index],
                        vector_index,
                    )
                }
            };

            let lpf = if pss.get_nstream() >= 3 {
                (0..nlpf)
                    .into_iter()
                    .map(|vector_index| get_parameter(2, vector_index))
                    .collect()
            } else {
                vec![]
            };
            let spectrum: Vec<f64> = (0..pss.get_vector_length(0))
                .into_iter()
                .map(|vector_index| get_parameter(0, vector_index))
                .collect();

            v.synthesize(
                get_parameter(1, 0),
                &spectrum,
                nlpf,
                &lpf,
                alpha,
                beta,
                volume,
                &mut speech[i * fperiod..],
            );

            for j in 0..pss.get_nstream() {
                if !pss.is_msd(j) || pss.get_msd_flag(j, i) {
                    frame_skipped_index[j] += 1;
                }
            }
        }

        GStreamSet { speech }
    }

    pub fn get_total_frame(&self) -> usize {
        self.speech.len()
    }
    pub fn get_speech(&self, sample_index: usize) -> f64 {
        self.speech[sample_index]
    }
}
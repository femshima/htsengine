use crate::{constants::NODATA, pstream::ParameterStreamSet, vocoder::Vocoder};

pub struct GenerateSpeechStreamSet {
    speech: Vec<f64>,
}

impl GenerateSpeechStreamSet {
    /// Generate speech
    pub fn create(pss: &ParameterStreamSet, mut v: Vocoder, fperiod: usize) -> Self {
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
        let mut frame_skipped_index = vec![0; pss.get_nstream()];
        for i in 0..total_frame {
            let get_parameter = |stream_index: usize, vector_index: usize| {
                if !pss.get_msd_flag(stream_index, i) {
                    NODATA
                } else {
                    pss.get_parameter(
                        stream_index,
                        frame_skipped_index[stream_index],
                        vector_index,
                    )
                }
            };

            let lpf = if pss.get_nstream() >= 3 {
                (0..pss.get_vector_length(2))
                    .map(|vector_index| get_parameter(2, vector_index))
                    .collect()
            } else {
                vec![]
            };
            let spectrum: Vec<f64> = (0..pss.get_vector_length(0))
                .map(|vector_index| get_parameter(0, vector_index))
                .collect();

            v.synthesize(
                get_parameter(1, 0),
                &spectrum,
                &lpf,
                &mut speech[i * fperiod..(i + 1) * fperiod],
            );

            for (j, index) in frame_skipped_index.iter_mut().enumerate() {
                if pss.get_msd_flag(j, i) {
                    *index += 1;
                }
            }
        }

        GenerateSpeechStreamSet { speech }
    }

    /// Get synthesized speech waveform
    pub fn get_speech(&self) -> &[f64] {
        &self.speech
    }
}

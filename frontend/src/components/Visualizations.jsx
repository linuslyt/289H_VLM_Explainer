import { Box, CircularProgress, Typography } from '@mui/material';
import { isEmpty } from 'lodash';
import VisualizerColumn from './VisualizerColumn';

// TODO: stack columns if screen width too small

const Visualizations = ({ selectedToken, isExplaining, uploadedImgPath, explanationData }) => {
  const imageUploaded = !isEmpty(uploadedImgPath);

  let render = <></>
  if (!selectedToken) {
    if (!imageUploaded) { // Image not yet uploaded
      render = (<Box display="flex" flex={1} alignItems="center" justifyContent="center" flexDirection="column" color="text.secondary">
            <Typography variant="h6">Upload an image to get started.</Typography>
          </Box>)
    } else { // Image uploaded, token not selected
      render = (<Box display="flex" flex={1} alignItems="center" justifyContent="center" flexDirection="column" color="text.secondary">
                  <Typography variant="h6">Select token from generated caption for analysis.</Typography>
                </Box>)
    }
  } else { // Token selected
    if (isExplaining) { // importance calculations in progress
      render = (<Box flex={1} display="flex" alignItems="center" justifyContent="center">
                    <CircularProgress />
                </Box>)
    } else if (isEmpty(explanationData)) { // error occurred
      render = (<Box display="flex" flex={1} alignItems="center" justifyContent="center" flexDirection="column" color="text.secondary">
                  <Typography variant="h6">An error occurred. Check logs for details.</Typography>
                </Box>)
    } else { // data retrieved
      render = (
        <Box display="flex" flex={1} overflow="hidden">
          {/* Global Activations */}
          <VisualizerColumn 
              title="Concept Activations"
              subtitle="What is present in the internal state?"
              data={{
                scores: explanationData.activations,
                concept_order: explanationData.indices_by_activations,
                text_groundings: explanationData.text_groundings,
                image_groundings: explanationData.image_grounding_paths,
              }}
              type="activations"
              color="#2196f3"
              bgColor="#e3f2fd"
              borderColor="#bbdefb"
          />

          {/* Local Importance */}
          <VisualizerColumn 
              title="Concept Importance Scores"
              subtitle="What actually caused this prediction?"
              data={{
                scores: explanationData.importance_scores,
                concept_order: explanationData.indices_by_importance,
                text_groundings: explanationData.text_groundings,
                image_groundings: explanationData.image_grounding_paths,
              }}
              type="importance"
              color="#4caf50"
              bgColor="#e8f5e9"
              borderColor="#c8e6c9"
          />
        </Box>
      )
    }
  }

  return (
    <Box sx={{ 
      width: '60%', 
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      bgcolor: '#fafafa',
      borderLeft: '1px solid #e0e0e0',
      position: 'relative'
    }}>
      {render}
    </Box>
  );
};

export default Visualizations;

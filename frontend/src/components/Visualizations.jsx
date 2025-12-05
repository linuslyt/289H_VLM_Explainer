import { Box, CircularProgress, Typography } from '@mui/material';
import VisualizerColumn from './VisualizerColumn';

// TODO: stack columns if screen width too small

const Visualizations = ({ selectedToken, isExplaining, explanationData }) => {
  return (
    <Box sx={{ 
      width: '50%', 
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      bgcolor: '#fafafa',
      borderLeft: '1px solid #e0e0e0',
      position: 'relative'
    }}>
       {!selectedToken ? (
         <Box display="flex" flex={1} alignItems="center" justifyContent="center" flexDirection="column" color="text.secondary">
            <Typography variant="h6">Select a token to analyze</Typography>
            <Typography variant="body2">Click on a word in the caption on the left</Typography>
         </Box>
       ) : (
         <Box display="flex" flexDirection="column" height="100%">
           {/* Header for Right Panel */}
           <Box p={2} borderBottom="1px solid #e0e0e0" bgcolor="white">
             <Typography variant="h6">
               Explanation for: <Box component="span" color="primary.main" fontWeight="bold">"{selectedToken}"</Box>
             </Typography>
           </Box>

           {/* Comparison Columns */}
           {isExplaining || !explanationData ? (
               <Box flex={1} display="flex" alignItems="center" justifyContent="center">
                   <CircularProgress />
               </Box>
           ) : (
           <Box display="flex" flex={1} overflow="hidden">
             {/* Global Activations */}
             <VisualizerColumn 
                title="Global Activation (CoX-LMM)"
                subtitle="What is present in the internal state?"
                data={explanationData.global_activations}
                type="global"
                color="#2196f3"
                bgColor="#e3f2fd"
                borderColor="#bbdefb"
             />

             {/* Local Importance */}
             <VisualizerColumn 
                title="Local Importance (Gradient)"
                subtitle="What actually caused this prediction?"
                data={explanationData.local_importances}
                type="local"
                color="#4caf50"
                bgColor="#e8f5e9"
                borderColor="#c8e6c9"
             />
           </Box>
           )}
         </Box>
       )}
    </Box>
  );
};

export default Visualizations;

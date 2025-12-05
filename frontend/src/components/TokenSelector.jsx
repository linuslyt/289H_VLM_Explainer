import { Box, CircularProgress, Typography } from '@mui/material';

const TokenSelector = ({ caption, selectedToken, isProcessing, onTokenClick }) => {
  return (
    <Box sx={{ 
      height: '15%', // Adjusted slightly to ensure text fits comfortably
      p: 2, 
      display: 'flex', 
      flexDirection: 'column',
      bgcolor: '#fff'
    }}>
       <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="overline" color="text.secondary" fontWeight="bold">
            Generated Caption
          </Typography>
          {selectedToken && (
             <Typography variant="caption" sx={{ bgcolor: 'primary.main', color: 'white', px: 1, borderRadius: 1 }}>
               Token to analyze: {selectedToken}
             </Typography>
          )}
       </Box>

       <Box sx={{ 
         flex: 1, 
         display: 'flex', 
         alignItems: 'center', 
         justifyContent: 'center',
         flexWrap: 'wrap',
         gap: 1
       }}>
          {isProcessing ? (
              <CircularProgress size={24} />
          ) : !caption ? (
            <Typography variant="body2" color="text.disabled" fontStyle="italic">
              Upload image for captioning to begin...
            </Typography>
          ) : (
            // remove punctuation then split by 
            caption
              .replace(/[^\w\s]|_/g, '') // Remove punctuation
              .replace(/\s+/g, " ") // Replace multiple spaces with one
              .trim() // Remove leading/trailing spaces
              .split(' ') // Split by single space
              .map((word, i) => (
              <Typography
                key={i}
                onClick={() => onTokenClick(word)}
                sx={{
                  cursor: 'pointer',
                  px: 0.8,
                  py: 0.4,
                  borderRadius: 1,
                  fontSize: '1.1rem',
                  border: selectedToken === word ? '1px solid' : '1px solid transparent',
                  borderColor: 'primary.main',
                  bgcolor: selectedToken === word ? 'primary.light' : 'transparent',
                  color: selectedToken === word ? 'white' : 'text.primary',
                  transition: 'all 0.1s',
                  '&:hover': { bgcolor: 'grey.200', color: 'black' }
                }}
              >
                {word}
              </Typography>
            ))
          )}
       </Box>
    </Box>
  );
};

export default TokenSelector;

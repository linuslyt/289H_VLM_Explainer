import { Box, Card, CardContent, Chip, Stack, Typography } from '@mui/material';
import ConceptBar from './ConceptBar';

/*
concept format:
{
  score: activation or importance score,
  images: image groundings,
  keywords: text groundings,
}
*/

const ConceptCard = ({ concept, type }) => {
  // Type determines color: 'global' -> Blue, 'local' -> Green
  // TODO: pairs for negative
  const color = type === 'activations' ? '#2196f3' : '#4caf50';
  const label = type === 'importance' ? 'Actv' : 'Impt';

  return (
    <Card variant="outlined" sx={{ mb: 2, borderRadius: 2, borderColor: '#eeeeee' }}>
      <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
          <Typography variant="caption" fontWeight="bold">
            Concept {concept.concept_id}
          </Typography>
          <ConceptBar score={concept.score} color={color} label={label} />
        </Box>

        <Typography variant="caption" color="text.secondary" display="block" mb={0.5}>
          Image groundings
        </Typography>
        <Stack 
          direction="row" 
          spacing={1} 
          mb={1.5}
          sx={{
            // Enable horizontal scrolling when too many images
            overflowX: 'auto',
            // Padding to prevent scrollbar from covering bottom of images
            pb: 1, 
            scrollbarWidth: 'thin',
            '&::-webkit-scrollbar': { height: '6px' },
            '&::-webkit-scrollbar-thumb': { 
              backgroundColor: 'rgba(0,0,0,0.2)', 
              borderRadius: '3px' 
            }
          }}
        >
          {concept.images.map((img, idx) => (
            <Box 
              key={idx}
              component="a" // make image clickable to open in new window
              href={img}
              target="_blank"
              rel="noopener noreferrer" 
              sx={{ 
                // prevent images from squishing when scrolling kicks in
                flexShrink: 0, 
                textDecoration: 'none',
                display: 'block' // remove default inline-block anchor spacing
              }}
            >
              <Box 
                component="img"
                src={img}
                alt={`Visual grounding ${idx}`}
                sx={{
                  width: 48, 
                  height: 48, 
                  borderRadius: 1, 
                  objectFit: 'cover',
                  bgcolor: '#f0f0f0',
                  // 6. Visual feedback
                  cursor: 'pointer',
                  transition: 'opacity 0.2s',
                  '&:hover': { opacity: 0.8 }
                }}
              />
            </Box>
          ))}
        </Stack>

        <Typography variant="caption" color="text.secondary" display="block" mb={0.5}>
          Text groundings
        </Typography>
        <Box display="flex" flexWrap="wrap" gap={0.5}>
          {concept.keywords.map((kw, idx) => (
            <Chip 
              key={idx} 
              label={kw} 
              size="small" 
              sx={{ 
                height: 20, 
                fontSize: '0.7rem', 
                bgcolor: `${color}15`, // very light opacity
                color: color,
                border: `1px solid ${color}40`
              }} 
            />
          ))}
        </Box>
      </CardContent>
    </Card>
  );
};

export default ConceptCard;

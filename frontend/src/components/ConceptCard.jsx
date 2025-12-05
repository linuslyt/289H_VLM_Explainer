import { Box, Card, CardContent, Chip, Stack, Typography } from '@mui/material';
import ConceptBar from './ConceptBar';

const ConceptCard = ({ concept, type, maxScore = 1, color_scheme}) => {
  // 1. Determine Color based on type and score sign
  const scheme = color_scheme[type] || color_scheme.activations;
  const color = concept.score >= 0 ? scheme.pos : scheme.neg;
  
  // 2. Determine Label
  const label = type === 'importance' ? 'Impt' : 'Actv';

  return (
    <Card variant="outlined" sx={{ mb: 2, borderRadius: 2, borderColor: '#eeeeee' }}>
      <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
        
        {/* Header: ID and Bar */}
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
          <Typography variant="caption" fontWeight="bold" flexShrink={0}>
            Concept {concept.concept_id}
          </Typography>
          
          {/* Passed maxScore for relative scaling.
            The bar now flexes to fill available space in the row.
          */}
          <ConceptBar 
            score={concept.score} 
            maxScore={maxScore} 
            color={color} 
            label={label} 
          />
        </Box>

        {/* Image Groundings */}
        <Typography variant="caption" color="text.secondary" display="block" mb={0.5}>
          Image groundings
        </Typography>
        <Stack 
          direction="row" 
          spacing={1} 
          mb={1.5}
          sx={{
            overflowX: 'auto',
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
              component="a" 
              href={img}
              target="_blank"
              rel="noopener noreferrer" 
              sx={{ 
                flexShrink: 0, 
                textDecoration: 'none',
                display: 'block' 
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
                  cursor: 'pointer',
                  transition: 'opacity 0.2s',
                  '&:hover': { opacity: 0.8 }
                }}
              />
            </Box>
          ))}
        </Stack>

        {/* Text Groundings */}
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
                bgcolor: `${color}15`, // very light opacity using hex alpha
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
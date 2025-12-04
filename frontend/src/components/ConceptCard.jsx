import React from 'react';
import { Box, Card, CardContent, Typography, Chip, Stack } from '@mui/material';
import ConceptBar from './ConceptBar';

const ConceptCard = ({ concept, type }) => {
  // Type determines color: 'global' -> Blue, 'local' -> Green
  const color = type === 'global' ? '#2196f3' : '#4caf50';
  const label = type === 'global' ? 'Actv' : 'Impt';

  return (
    <Card variant="outlined" sx={{ mb: 2, borderRadius: 2, borderColor: '#eeeeee' }}>
      <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
          <Typography variant="subtitle2" fontWeight="bold">
            {concept.id}
          </Typography>
          <ConceptBar score={concept.score} color={color} label={label} />
        </Box>

        <Typography variant="caption" color="text.secondary" display="block" mb={0.5}>
          Representative Images
        </Typography>
        <Stack direction="row" spacing={1} mb={1.5}>
          {concept.images.map((img, idx) => (
            <Box 
              key={idx}
              component="img"
              src={img}
              alt={`Visual grounding ${idx}`}
              sx={{ 
                width: 48, 
                height: 48, 
                borderRadius: 1, 
                objectFit: 'cover',
                bgcolor: '#f0f0f0'
              }}
            />
          ))}
        </Stack>

        <Typography variant="caption" color="text.secondary" display="block" mb={0.5}>
          Associated Keywords
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

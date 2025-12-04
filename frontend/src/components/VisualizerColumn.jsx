import React from 'react';
import { Box, Typography } from '@mui/material';
import ConceptChart from './ConceptChart';
import ConceptCard from './ConceptCard';

const VisualizerColumn = ({ title, subtitle, data, type, color, bgColor, borderColor }) => {
  return (
    <Box flex={1} display="flex" flexDirection="column" borderRight="1px solid #e0e0e0">
      <Box p={2} bgcolor={bgColor} borderBottom={`1px solid ${borderColor}`}>
        <Typography variant="subtitle1" fontWeight="bold" color={color}>
          {title}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {subtitle}
        </Typography>
      </Box>
      <Box px={2} pt={2} borderBottom="1px solid #eee" bgcolor="#f1f8ff">
          <ConceptChart data={data} color={color} />
      </Box>
      <Box flex={1} p={2} sx={{ overflowY: 'auto' }}>
        {data.map((concept, idx) => (
          <ConceptCard key={idx} concept={concept} type={type} />
        ))}
      </Box>
    </Box>
  );
};

export default VisualizerColumn;

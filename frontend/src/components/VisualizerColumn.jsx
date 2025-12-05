import { Box, Typography } from '@mui/material';
import isEmpty from 'lodash/isEmpty';
import ConceptCard from './ConceptCard';
import ConceptChart from './ConceptChart';

// data format
// {
//   scores: activations or importance score, sorted by concept indices,
//   concept_order: concept indices sorted by score,
//   text_groundings: text groundings,
//   image_groundings: image grounding paths,
// }

// concept card format
// {
//   score: activation or importance score,
//   images: image groundings,
//   keywords: text groundings,
//   concept_id: concept index in dictionary + 1,
// }

// concept chart format
// {
//   score: activation or importance score,
//   concept_id: concept index in dictionary + 1,
// }
const VisualizerColumn = ({ title, subtitle, data, type, color, bgColor, borderColor }) => {
  // element of data.concept_order = concept index in dictionary
  // index of data.concept_order = order of concept by score
  if (isEmpty(data)) return;
  console.log(data)

  let concept_chart_data = data.scores.map((score, concept_index) => { // map (element, index)
    return {
      score: score,
      concept_id: concept_index + 1,
    };
  })

  let concept_card_data = data.concept_order.map((concept_index, concept_order) => { // map (element, index)
    return {
      score: data.scores[concept_index],
      keywords: data.text_groundings[concept_index],
      images: data.image_groundings[concept_index].map(img_path => `http://localhost:8000/grounding-images/${img_path}`).flatMap(item => Array(20).fill(item)), // for testing overflow
      concept_id: concept_index + 1,
    };
  })

  return (
      <Box 
        flex={1} 
        minWidth={0} 
        display="flex" 
        flexDirection="column" 
        borderRight="1px solid #e0e0e0"
      >
        <Box p={2} bgcolor={bgColor} borderBottom={`1px solid ${borderColor}`}>
          <Typography variant="subtitle1" fontWeight="bold" color={color}>
            {title}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {subtitle}
          </Typography>
        </Box>
        
        <Box px={2} pt={2} borderBottom="1px solid #eee" bgcolor="#f1f8ff">
            <ConceptChart data={concept_chart_data} color={color} type={type} />
        </Box>

        <Box flex={1} p={2} sx={{ overflowY: 'auto' }}>
          {concept_card_data.map((concept, idx) => (
            <ConceptCard key={idx} concept={concept} type={type} />
          ))}
        </Box>
      </Box>
    );
};

export default VisualizerColumn;

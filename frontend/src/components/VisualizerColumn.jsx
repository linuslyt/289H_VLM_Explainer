import { Box, Typography } from '@mui/material';
import isEmpty from 'lodash/isEmpty';
import { useRef } from 'react';
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
  // Store refs to card elements: refs.current[concept_id] = DOMNode
  // Clicking on bar will scroll to card
  const cardRefs = useRef({});

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

  // Define color schemes
  const COLOR_SCHEME = {
    activations: {
      pos: "#2196f3", // Blue
      neg: "#ef5350"  // Red
    },
    importance: {
      pos: "#4caf50", // Green
      neg: "#ff9800"  // Orange
    }
  };

  // Scroll Handler
  const handleBarClick = (conceptId) => {
    const element = cardRefs.current[conceptId];
    if (element) {
      element.scrollIntoView({ 
        behavior: 'smooth', 
        block: 'start' // Aligns top of card to top of scroll area
      });
      
      // Add a temporary highlight effect
      element.style.transition = "background-color 0.5s";
      const originalBg = element.style.backgroundColor;
      element.style.backgroundColor = "#fffde7"; // Light yellow highlight
      setTimeout(() => {
        element.style.backgroundColor = originalBg;
      }, 1000);
    }
  };

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
            <ConceptChart 
              data={concept_chart_data} 
              color={color} 
              type={type} 
              color_scheme={COLOR_SCHEME}
              onBarClick={handleBarClick} 
            />
        </Box>

        <Box flex={1} p={2} sx={{ overflowY: 'auto' }}>
          {concept_card_data.map((concept, idx) => (
            <ConceptCard 
              key={idx} 
              concept={concept} 
              type={type} 
              color_scheme={COLOR_SCHEME}
              // Register the ref for this specific concept ID
              domRef={(el) => (cardRefs.current[concept.concept_id] = el)}
            />
          ))}
        </Box>
      </Box>
    );
};

export default VisualizerColumn;
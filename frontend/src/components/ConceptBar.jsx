import { Box, Typography } from '@mui/material';
import * as d3 from 'd3';
import { useEffect, useRef } from 'react';

const ConceptBar = ({ score, maxScore = 1, color, label }) => {
  const d3Container = useRef(null);

  useEffect(() => {
    if (d3Container.current && score !== null) {
      const height = 20;
      // Measure the actual pixel width of the container to determine text contrast
      const currentWidth = d3Container.current.clientWidth || 0;
      
      const svg = d3.select(d3Container.current);

      // Clear previous
      svg.selectAll("*").remove();

      // Set SVG to be responsive
      svg.attr("width", "100%")
         .attr("height", height);

      // Background Track (Gray)
      svg.append("rect")
         .attr("width", "100%")
         .attr("height", height)
         .attr("fill", "#f5f5f5") 
         .attr("rx", 4);

      // Calculate width percentage based on maxScore
      const safeMax = maxScore === 0 ? 1 : maxScore;
      const ratio = Math.min(1, Math.abs(score) / safeMax);
      const percentage = ratio * 100;

      // Determine Text Color
      // "0.000" at 10px font is approx 30-35px wide. +6px padding = ~40px.
      // If the colored bar is narrower than 40px, the text sits on the light gray background.
      const barPixelWidth = currentWidth * ratio;
      const textColor = barPixelWidth < 35 ? "#424242" : "#ffffff";

      // Foreground Bar (Colored)
      const bar = svg.append("rect")
         .attr("height", height)
         .attr("fill", color)
         .attr("rx", 4)
         .attr("width", 0); // Start at 0 for animation

      // Animate to final width
      bar.transition()
         .duration(800)
         .attr("width", `${percentage}%`);
      
      // Text label inside bar
      svg.append("text")
         .attr("x", 6)
         .attr("y", height / 2)
         .attr("dy", ".35em")
         .attr("fill", textColor) // Apply calculated contrast color
         .attr("font-size", "10px")
         .attr("font-weight", "bold")
         .style("pointer-events", "none") 
         .text(score.toFixed(3)); 
    }
  }, [score, maxScore, color]);

  return (
    <Box display="flex" alignItems="center" gap={1} flex={1} ml={2} minWidth={0}>
      <Typography variant="caption" width={30} fontWeight="bold" color="text.secondary" flexShrink={0}>
        {label}
      </Typography>
      <Box flex={1} display="flex" alignItems="center">
        <svg ref={d3Container} style={{ width: '100%', display: 'block' }} />
      </Box>
    </Box>
  );
};

export default ConceptBar;
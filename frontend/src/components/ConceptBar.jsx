import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { Box, Typography } from '@mui/material';

const ConceptBar = ({ score, color, label }) => {
  const d3Container = useRef(null);

  useEffect(() => {
    if (d3Container.current && score !== null) {
      const width = 200;
      const height = 20;
      const svg = d3.select(d3Container.current);

      // Clear previous
      svg.selectAll("*").remove();

      svg.attr("width", width)
         .attr("height", height);

      // Background
      svg.append("rect")
         .attr("width", width)
         .attr("height", height)
         .attr("fill", "#e0e0e0")
         .attr("rx", 5);

      // Scale
      const xScale = d3.scaleLinear()
                       .domain([0, 1])
                       .range([0, width]);

      // Foreground Bar
      svg.append("rect")
         .attr("width", xScale(score))
         .attr("height", height)
         .attr("fill", color)
         .attr("rx", 5)
         .transition()
         .duration(1000)
         .attr("width", xScale(score));
      
      // Text label inside bar (optional)
      svg.append("text")
         .attr("x", 5)
         .attr("y", height / 2)
         .attr("dy", ".35em")
         .attr("fill", "white")
         .attr("font-size", "10px")
         .attr("font-weight", "bold")
         .text(score.toFixed(2));

    }
  }, [score, color]);

  return (
    <Box display="flex" alignItems="center" gap={1}>
      <Typography variant="caption" width={40} fontWeight="bold" color="text.secondary">{label}</Typography>
      <svg ref={d3Container} />
    </Box>
  );
};

export default ConceptBar;

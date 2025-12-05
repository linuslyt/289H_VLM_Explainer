import { Box, Typography } from '@mui/material';
import * as d3 from 'd3';
import { useEffect, useRef } from 'react';

const ConceptChart = ({ data, color = "#2196f3" }) => {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!data || data.length === 0) return;

    const margin = { top: 10, right: 10, bottom: 30, left: 30 };
    // Make chart responsive to container, but for D3 calculations we need a fixed or relative size.
    // We'll rely on viewBox for scaling.
    const width = 300 - margin.left - margin.right;
    const height = 150 - margin.top - margin.bottom;

    // Clear previous render
    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3.select(svgRef.current)
      .attr("width", "100%")
      .attr("height", "100%")
      .attr("viewBox", `0 0 300 150`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // X axis
    const x = d3.scaleBand()
      .rangeRound([0, width])
      .padding(0.2);

    const y = d3.scaleLinear()
      .rangeRound([height, 0]);

    x.domain(data.map(d => d.concept_id));
    y.domain(d3.extent(data.map(d => d.score)));

    // Add X Axis
    svg.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x).tickSize(0))
      .selectAll("text")
      .attr("transform", "translate(0, 5)")
      .style("font-size", "10px")
      .style("text-anchor", "middle");

    // Add Y Axis
    svg.append("g")
      .call(d3.axisLeft(y).ticks(4).tickSize(-width))
      .call(g => g.select(".domain").remove())
      .selectAll(".tick line")
      .attr("stroke", "#eee")
      .attr("stroke-dasharray", "2,2");
    
    svg.selectAll(".tick text")
       .style("font-size", "9px")
       .style("fill", "#999");

    // Add Bars
    svg.selectAll(".bar")
      .data(data)
      .enter().append("rect")
      .attr("class", "bar")
      .attr("x", d => x(d.concept_id))
      .attr("y", d => y(d.score))
      .attr("width", x.bandwidth())
      .attr("height", d => height - y(d.score))
      .attr("fill", color);

  }, [data, color]);

  return (
    <Box sx={{ width: '100%', height: '150px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
       {(!data || data.length === 0) ? (
           <Typography variant="caption" color="text.secondary">No data</Typography>
       ) : (
           <svg ref={svgRef} style={{ width: '100%', height: '100%' }}></svg>
       )}
    </Box>
  );
};

export default ConceptChart;
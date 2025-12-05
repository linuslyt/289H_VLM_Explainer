import { Box, Typography } from '@mui/material';
import * as d3 from 'd3';
import { useEffect, useRef } from 'react';

const ConceptChart = ({ data, type = "activations", color_scheme, onBarClick }) => {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!data || data.length === 0) return;
    const margin = { top: 20, right: 10, bottom: 45, left: 50 };
    
    // ViewBox dimensions
    const vbWidth = 350;
    const vbHeight = 200;
    const width = vbWidth - margin.left - margin.right;
    const height = vbHeight - margin.top - margin.bottom;

    // Clear previous render
    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3.select(svgRef.current)
      .attr("width", "100%")
      .attr("height", "100%")
      .attr("viewBox", `0 0 ${vbWidth} ${vbHeight}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // --- SCALES ---
    const x = d3.scaleBand()
      .rangeRound([0, width])
      .padding(0.2);

    const y = d3.scaleLinear()
      .rangeRound([height, 0]);

    // X Domain: Concept IDs
    x.domain(data.map(d => d.concept_id));

    // Y Domain: Extent of data, but ALWAYS including 0 to handle negatives properly
    const minVal = d3.min(data, d => d.score);
    const maxVal = d3.max(data, d => d.score);
    y.domain([
      Math.min(0, minVal), 
      Math.max(0, maxVal)
    ]).nice();

    const y0 = y(0); // The pixel position of the "0" line

    // --- AXES ---
    // X Axis (placed at the bottom of the chart area)
    svg.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x).tickSize(0)) 
      .selectAll("text")
      .attr("transform", "translate(0, 5)")
      .style("font-size", "10px")
      .style("text-anchor", "middle");

    // X Axis Title
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", height + 30) // Position below axis
      .style("text-anchor", "middle")
      .style("font-size", "12px")
      .style("fill", "#666")
      .text("Concept ID");

    // Y Axis
    svg.append("g")
      .call(d3.axisLeft(y).ticks(5).tickSize(-width)) // Gridlines across full width
      .call(g => g.select(".domain").remove()) // Remove the vertical axis line
      .selectAll(".tick line")
      .attr("stroke", "#acacacff")
      .attr("stroke-dasharray", "2,2");
    
    svg.selectAll(".tick text")
       .style("font-size", "10px")
       .style("fill", "#999");

    // Y Axis Title
    svg.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -40) // Position left of axis
      .attr("x", -height / 2)
      .style("text-anchor", "middle")
      .style("font-size", "12px")
      .style("fill", "#666")
      .text(type === "activations" ? "Activations" : "Importance");

    // --- ZERO LINE ---
    svg.append("line")
      .attr("x1", 0)
      .attr("x2", width)
      .attr("y1", y0)
      .attr("y2", y0)
      .attr("stroke", "#333")
      .attr("stroke-width", 1);

    // --- BARS ---
    // Get current colors based on type, fallback to activations if type is invalid
    const colors = color_scheme[type] || color_scheme.activations;

    svg.selectAll(".bar")
      .data(data)
      .enter().append("rect")
      .attr("class", "bar")
      .attr("x", d => x(d.concept_id))
      .attr("width", x.bandwidth())
      // Logic for Y position and Height based on positive/negative
      .attr("y", d => d.score >= 0 ? y(d.score) : y0)
      .attr("height", d => Math.abs(y(d.score) - y0))
      // Logic for Color
      .attr("fill", d => d.score >= 0 ? colors.pos : colors.neg)
      
      // --- INTERACTION ---
      .style("cursor", "pointer") // Visual feedback
      .on("click", (event, d) => {
        if (onBarClick) {
          onBarClick(d.concept_id);
        }
      })
      
      // Tooltip logic
      .append("title")
      .text(d => `Concept ${d.concept_id}: ${d.score.toFixed(6)}`);

  }, [data, type, onBarClick]);

  return (
    <Box sx={{ width: '100%', height: '200px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
       {(!data || data.length === 0) ? (
           <Typography variant="caption" color="text.secondary">No data</Typography>
       ) : (
           <svg ref={svgRef} style={{ width: '100%', height: '100%' }}></svg>
       )}
    </Box>
  );
};

export default ConceptChart;
import React, { useEffect, useRef } from "react"
import {
  withStreamlitConnection,
  Streamlit,
  ComponentProps,
} from "streamlit-component-lib"
import * as d3 from "d3"

const D3Events = ({ args }: ComponentProps) => {
  //Destructure props as seen in Streamlit docs-------------------------------------------------------------------
  const predictionData: [number, number][] = args["data"]
  const referenceData: [number, number][] = args["referenceFunction"]

  //Boilerplate for referencing html element for d3 and setting margins and graph size-----------------------------
  const svgRef = useRef(null)
  let graphWidth = 700
  let graphHeight = 400
  var margin = { top: 20, right: 30, bottom: 20, left: 30 },
    width = graphWidth - margin.left - margin.right,
    height = graphHeight - margin.top - margin.bottom

  useEffect(() => {
    render()
    return () => {
      d3.selectAll("svg > *").remove()
    }
  }, [predictionData, referenceData])

  useEffect(() => {
    Streamlit.setFrameHeight()
  }, [graphHeight])

  //Function that uses d3 to to construct and render svg elements-----------------------------------------------//
  const render = () => {
    //Create graph and reference svg html element-----------------------------------------------------------------
    var svg = d3
      .select(svgRef.current)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")")

    //Append/Style axises and functions to convert a value on x/y axises to absolute pixel position--------------
    var x = d3.scaleLinear().domain([0, 10]).range([0, width])
    svg
      .append("g")
      .attr("transform", "translate(0," + height + ")")
      .style("font-size", 12 + "px")
      .call(d3.axisBottom(x))
    var y = d3.scaleLinear().domain([-1, 1]).range([height, 0])
    svg
      .append("g")
      .style("font-size", 12 + "px")
      .call(d3.axisLeft(y))
    svg
      .selectAll(".domain")
      .attr("stroke", "#E0E2E4")
      .attr("stroke-width", 1 + "px")
    svg.selectAll(".tick line").attr("opacity", "0")
    svg.selectAll(".tick text").style("fill", "#E0E2E4")

    //Append/Style curve whose points correspond to PREDICTION data-----------------------------------------------
    svg
      .append("path")
      .datum(predictionData)
      .attr("id", "predictionCurve")
      .attr("fill", "none")
      .attr("stroke", "red")
      .attr("stroke-width", 1.5)
      .attr(
        "d",
        d3
          .line()
          .curve(d3.curveLinear)
          .x((d) => x(d[0]))
          .y((d) => y(d[1]))
      )
      .style("stroke-width", "2px")

    //Append/Style curve whose points correspond to REFERENCE data-----------------------------------------------
    svg
      .append("path")
      .datum(referenceData)
      .attr("id", "referenceCurve")
      .attr("fill", "none")
      .attr("stroke", "white")
      .attr("stroke-width", 1.5)
      .attr(
        "d",
        d3
          .line()
          .curve(d3.curveBasis)
          .x((d) => x(d[0]))
          .y((d) => y(d[1]))
      )
      .style("stroke-dasharray", "4")
      .style("stroke-width", "2px")
  }

  return (
    <div className="justify-center">
      <svg
        version="1.1"
        x="0px"
        y="0px"
        width={`${graphWidth}px`}
        height={`${graphHeight}px`}
        ref={svgRef}
        viewBox={`0 0 ${graphWidth} ${graphHeight}`}
        preserveAspectRatio="xMaxYMid meet"
      >
        <svg ref={svgRef} />
      </svg>
    </div>
  )
}

export default withStreamlitConnection(D3Events)

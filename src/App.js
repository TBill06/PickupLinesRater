import React, { useState } from "react";
import axios from "axios";
import { Bar } from "react-chartjs-2";
import "chart.js/auto";
import './App.css';

const RatingApp = () => {
  const [selectedSystem, setSelectedSystem] = useState("");
  const [pickupLine, setPickupLine] = useState("");
  const [ratings, setRatings] = useState({});
  const [chartData, setChartData] = useState(null);

  const systems = ["BoW", "VADER", "Naive Bayes", "Markov Chains", "BERT", "GPT"];
  const displayOrder = {
    "Weak": ["BoW", "VADER"],
    "Mid": ["Markov Chains", "Naive Bayes"],
    "Strong": ["BERT", "GPT"]
  };

  const pickupLines = [
    "Are you a magician? Because whenever I look at you, everyone else disappears.",
    "Do you believe in love at first sight, or should I walk by again?",
    "Are you a Wi-Fi signal? Because I’m feeling a strong connection.",
    "Please pass me the salt.",
    "The boiling point of water is 100 degrees Celsius.",
    "Are you French? Because Eiffel for you!",
    "I need directions to your heart.",
    "You are amazing and beautiful!",
    "Did it hurt when you fell from heaven?",
    "You’re so sweet; I might just get a cavity talking to you!",
    "If I had a nickel for every time I thought about you, I’d be a millionaire and buy you the world.",
    "Do you have a map? I keep getting lost in your eyes.",
    "Is your name Google? Because you have everything I’ve been searching for.",
    "You must be tired because you’ve been running through my mind all day.",
    "Are you a parking ticket? Because you’ve got FINE written all over you.",
    "Do you believe in love at first sight, or should I walk by again?",
    "I must be a snowflake, because I’ve fallen for you.",
    "If you were a vegetable, you’d be a cute-cumber.",
    "You remind me of my ex—complicated and unforgettable.",
    "You’re okay, I guess."
  ];

  const fetchRating = async () => {
    if (!selectedSystem || !pickupLine) return alert("Enter a line and system!");
    try {
      const response = await axios.post("http://127.0.0.1:5001/api/rate", {
        system: selectedSystem,
        line: pickupLine,
      });
      setRatings({ [selectedSystem]: response.data.rating.toFixed(3) });
      setChartData(null); // Clear chart data
    } catch (error) {
      console.error("Error fetching rating:", error);
    }
  };

  const fetchAllRatingsAndPlot = async () => {
    if (!pickupLine) return alert("Enter a pickup line!");
    // Clear old data
    setRatings({});
    setChartData(null);
    try {
      const response = await axios.post("http://127.0.0.1:5001/api/rate-all", { line: pickupLine });
      const formattedRatings = {};
      const labels = [];
      const data = [];
      for (const [system, rating] of Object.entries(response.data.ratings)) {
        formattedRatings[system] = rating.toFixed(3);
      }
      setRatings(formattedRatings);

      for (const category of Object.keys(displayOrder)) {
        for (const system of displayOrder[category]) {
          if (formattedRatings[system] !== undefined) {
            labels.push(system);
            data.push(formattedRatings[system]);
          }
        }
      }

      setChartData({
        labels,
        datasets: [
          {
            label: "Ratings",
            data,
            backgroundColor: "rgba(75, 192, 192, 0.6)",
          },
        ],
      });
    } catch (error) {
      console.error("Error fetching all ratings and plotting chart:", error);
    }
  };

  const handlePickupLineChange = (e) => {
    setPickupLine(e.target.value);
    // Clear old data when new input is entered
    setRatings({});
    setChartData(null);
  };

  const handlePickupLineSelect = (e) => {
    setPickupLine(e.target.value);
    // Clear old data when new input is selected
    setRatings({});
    setChartData(null);
  };

  return (
    <div className="App">
      <div className="App-header">
        <h1>Pickup Line Rating App</h1>
        <div className="input-group">
          <textarea
            placeholder="Enter your pickup line here..."
            value={pickupLine}
            onChange={handlePickupLineChange}
          />
        </div>
        <div className="dropdown-group">
          <select onChange={handlePickupLineSelect}>
            <option value="">Select a Pickup Line</option>
            {pickupLines.map((line, index) => (
              <option key={index} value={line}>
                {line}
              </option>
            ))}
          </select>
          <select
            value={selectedSystem}
            onChange={(e) => setSelectedSystem(e.target.value)}
          >
            <option value="">Select Rating System</option>
            {systems.map((system) => (
              <option key={system} value={system}>
                {system}
              </option>
            ))}
          </select>
        </div>
        <div className="button-group">
          <button onClick={fetchRating}>Get Rating</button>
          <button onClick={fetchAllRatingsAndPlot}>Get All Ratings and Plot</button>
        </div>
        <div className="ratings">
          <div>
            <strong>Pickup Line:</strong> {pickupLine}
          </div>
          {Object.keys(displayOrder).map((category) => (
            <div key={category}>
              <strong>{category}:</strong>
              {displayOrder[category].map((system) => (
                <div key={system}>
                  {system}: {ratings[system] !== undefined ? ratings[system] : ""}
                </div>
              ))}
            </div>
          ))}
        </div>
        {chartData && (
          <div className="chart">
            <Bar 
              data={chartData} 
              options={{
                scales: {
                  y: {
                    min: 0,
                    max: 1
                  }
                }
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default RatingApp;
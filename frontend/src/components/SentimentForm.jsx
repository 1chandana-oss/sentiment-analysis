

import { useState } from "react";

export default function SentimentForm() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      setResult(data.sentiment);
    } catch (err) {
      setResult("Error connecting to API");
    }
    setLoading(false);
  };

  const getBadgeColor = () => {
    if (result === "positive") return "bg-green-500";
    if (result === "negative") return "bg-red-500";
    if (result === "neutral") return "bg-gray-500";
    return "";
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-r from-blue-50 to-purple-50">
      <div className="w-full max-w-md bg-white rounded-3xl shadow-xl p-8">
        <h1 className="text-3xl font-bold text-center mb-6">📝 Sentiment Analyzer</h1>
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <textarea
            className="p-4 border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
            rows="5"
            placeholder="Type your text here..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <button
            type="submit"
            disabled={loading}
            className="relative bg-blue-500 text-white py-3 rounded-xl font-semibold hover:bg-blue-600 transition flex items-center justify-center"
          >
            {loading && (
              <svg
                className="animate-spin h-5 w-5 mr-2 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                ></path>
              </svg>
            )}
            {loading ? "Analyzing..." : "Analyze Sentiment"}
          </button>
        </form>

        {result && (
          <div className="mt-6 text-center">
            <span
              className={`text-white text-lg font-semibold px-4 py-2 rounded-full ${getBadgeColor()}`}
            >
              Sentiment: {result}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

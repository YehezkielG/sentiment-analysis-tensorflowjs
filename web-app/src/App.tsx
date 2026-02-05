import { useState, useEffect } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";

function padSequence(sequences:string[][],maxLen:number,padding = "post",truncating = "post",pad_value = 0) {
  return sequences.map((seq:string[]) => {
    if (seq.length > maxLen) {      
      if (truncating === "pre") {
        seq.splice(0, seq.length - maxLen);
      } else {
        seq.splice(maxLen, seq.length - maxLen);
      }
    }
    if (seq.length < maxLen) {
      const pad = [];
      for (let i = 0; i < maxLen - seq.length; i++) {
        pad.push(pad_value);
      }
      if (padding === "pre") {
        seq = pad.concat(seq);
      } else {
        seq = seq.concat(pad);
      }
    }
    return seq;
  });
}

function App() {
  const [inputText, setInputText] = useState("");
  const [model, setModel] = useState<tf.GraphModel | null>(null);
  const [wordIndex, setWordIndex] = useState<{ [key: string]: number }>({});
  const [score, setScore] = useState<number>(-1);

  useEffect(() => {
    const loadResources = async () => {
      try {
        // 1. Load Model dari folder public
        const model = await tf.loadGraphModel("/model/model.json");
        console.log("Model loaded from public folder!");

        // 2. Load Word Index (JSON biasa) menggunakan fetch
        const response = await fetch("/model/word_index.json");
        const wordIndex = await response.json();
        console.log("Word index loaded!");
        setModel(model);
        setWordIndex(wordIndex);
        return { model, wordIndex };
      } catch (error) {
        console.error("Gagal memuat resource AI:", error);
      }
    };
    loadResources();
  }, []);

  const handleChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputText(event.target.value);
  };

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (model && Object.keys(wordIndex).length > 0) {
      const inputWords = inputText
        .toLowerCase()
        .replace(/[\.,!?\(\)]/g, "")
        .split(" ");
      const score = predict(inputWords);
      setScore(score);
  }
}
function predict(inputText: string[]):number { 
  if (!model) {
    alert("Model belum dimuat");
    throw new Error("Model belum dimuat");
  }
    const sequence = inputText.map((word) => {
    const indexed = wordIndex[word];
    if (indexed === undefined) {
      return 1; //change to oov value
    }return indexed;
  }); // Melakukan padding
  const maxlen = 236;
  const padding = "post";
  const truncating = "post";

  const paddedSequence = padSequence([sequence], maxlen, padding, truncating);
  const score = tf.tidy(() => {
    const input = tf.tensor2d(paddedSequence, [1, maxlen]);
    const result = model.predict(input) as tf.Tensor;
    return result.dataSync()[0];
  });
  return score;
}

  return (
    <>
      <h2>Sentiment Analysis with Tensorflow JS</h2>
      <div>
        this project will demonstrate sentiment analysis using tensorflow js,
        react and typescript. the sentiment theme is based on movie reviews. the train data cave from 
        <a href="https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews" target="_blank" rel="noreferrer">
          {" "}IMDB Movie Review Dataset
        </a>
      </div>
      <form action="">
        <div className="container">
          <textarea
            name="review"
            id="review"
            className="textArea"
            onChange={handleChange}
          ></textarea>
          <div className="btnContainer">
              <input type="submit" className="submitBtn" onClick={handleSubmit} />
          </div>
        <div>
          {score >= 0 && (
            <div>
              Sentiment: {score >= 0.5 ? "Positive" : "Negative"} (Confidence:
              {((score >= 0.5 ? score : 1 - score) * 100).toFixed(2)}%)
            </div>  
          )}
        </div>
        </div>
      </form>
    </>
  );
}

export default App;
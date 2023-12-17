import { useState } from 'react'
import './App.css'
import YouTube from 'react-youtube'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <div>
      </div>
      <h1>Youtube Supporter</h1>
      <YouTube videoId="cEynsEWpXdA" />
      <div className="card">
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
      </div>
    </>
  )
}

export default App

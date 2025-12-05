import {
  Box,
  CssBaseline,
  Divider,
  ThemeProvider,
  createTheme
} from '@mui/material';
import { useCallback, useState } from 'react';
import ImageUploader from './components/ImageUploader';
import LogConsole from './components/LogConsole';
import TokenSelector from './components/TokenSelector';
import Visualizations from './components/Visualizations';

// Create a dark/clean theme or keep default. 
// Using default here but removing body margins via CssBaseline.
const theme = createTheme();

// TODO: initialize log with server /status response. disable upload until server online

function App() {
  // --- STATE ---
  const [logs, setLogs] = useState([]);
  const [preview, setPreview] = useState(null);
  const [uploadedImgPath, setUploadedImgPath] = useState(null);
  const [caption, setCaption] = useState(null);
  const [selectedToken, setSelectedToken] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [explanationData, setExplanationData] = useState(null);
  const [isExplaining, setIsExplaining] = useState(false);

  // Logging helper
  const addLog = (message) => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    setLogs((prev) => [...prev, `[${time}] ${message}`]);
  };

  // Image drag and drop handler
  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // show image preview
    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);

    // reset state
    setLogs([]);
    setCaption(null);
    setSelectedToken(null);
    setExplanationData(null);
    setIsProcessing(true);
    addLog(`System: File loaded - ${file.name}`);

    try {
      addLog(`Network: Uploading image ${file.name} to server...`);

      // We must use FormData to send files via POST
      const formData = new FormData();
      // 'file' here must match the parameter name in FastAPI: def save_image(file: ...)
      formData.append("file", file); 

      const uploadResponse = await fetch("http://localhost:8000/upload/", {
        method: "POST",
        body: formData,
      });

      if (!uploadResponse.ok) {
        addLog(`Network: failed to upload ${file.name}. Please retry.`);
        throw new Error(`Upload failed: ${uploadResponse.statusText}`);
      }
      const uploadResult = await uploadResponse.json();
      const serverFilePath = uploadResult.filename;
      setUploadedImgPath(serverFilePath);
      
      addLog(`Backend: Image received. Requesting caption for img=${serverFilePath}`);

      // /caption-image returns SSE stream. event_type="log" for logs, event_type="return" for returned caption.
      const captionUrl = `http://localhost:8000/caption-image?uploaded_img_path=${encodeURIComponent(serverFilePath)}`;
      const captionSource = new EventSource(captionUrl);

      captionSource.addEventListener("log", (e) => {
        addLog(`Backend: ${e.data}`);
      })

      captionSource.addEventListener("return", (e) => {
          addLog("System: Caption generated")
          setCaption(e.data);
          setIsProcessing(false);
          console.log(e.data);
          captionSource.close()
      })

      captionSource.onerror = (err) => {
        console.error("Captioning stream failed:", err);
        captionSource.close(); // Close to stop retry loop
        addLog("Error: Stream connection lost during captioning.");
        setIsProcessing(false);
      };
    } catch (e) {
      console.error(e);
      addLog(`Error: ${e.message}`);
    }
  }, []);


  // Token select handler
  const handleTokenClick = useCallback(async (token) => {
      setSelectedToken(token);
      setIsExplaining(true);
      setExplanationData(null);
      
      const baseExplainUrl = "http://localhost:8000/importance-estimation";

      const params = new URLSearchParams({
        uploaded_img_path: uploadedImgPath,
        token_of_interest: token,
        sampled_subset_size: 5000,
        sampling_inference_batch_size: 20, // TODO: add input for this
        n_concepts: 10, // TODO: add slider for this
        force_recompute: false // TODO: add button to toggle this. Also set to true if n_concepts or sampled_subset_size changes for the same token of interest.
      });

      const encodedExplainURL = `${baseExplainUrl}?${params.toString()}`;

      try {
        const explainSource = new EventSource(encodedExplainURL);

        explainSource.addEventListener("log", (e) => {
          addLog(`Backend: ${e.data}`);
        })

        explainSource.addEventListener("return", (e) => {
            addLog("System: Explanations generated")
            setExplanationData(JSON.parse(e.data));
            setIsExplaining(false);
            console.log(e.data);
            explainSource.close()
        })

        explainSource.onerror = (err) => {
          console.error("Stream failed:", err);
          explainSource.close(); // Close to stop retry loop
          addLog(`Backend: ERROR - ${err.data}`);
          setIsExplaining(false);
        };
      } catch (e) {
        console.error(e);
        addLog(`Error: ${e.message}`);
      }
  }, [uploadedImgPath]);


  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />

      <Box sx={{ 
        // fill entire screen without scrolling
        display: 'flex', 
        width: '100vw', 
        height: '100vh', 
        overflow: 'hidden',
        bgcolor: '#f4f6f8'
      }}>
        <Box sx={{ 
          width: '40%', 
          height: '100%', 
          display: 'flex', 
          flexDirection: 'column',
          borderRight: '1px solid #e0e0e0',
          bgcolor: 'white'
        }}>
          {/* Drag and drop image uploader */}
          <ImageUploader onDrop={onDrop} preview={preview} />

          <Divider />

          {/* Caption display, token selector */}
          <TokenSelector 
            caption={caption}
            selectedToken={selectedToken}
            isProcessing={isProcessing}
            onTokenClick={handleTokenClick}
          />

          <Divider />

          {/* Streamed logs display */}
          <LogConsole logs={logs} />
        </Box>

        {/* Concept visualizer (Right Panel) */}
        <Visualizations
          selectedToken={selectedToken}
          isExplaining={isExplaining}
          uploadedImgPath={uploadedImgPath}
          explanationData={explanationData}
        />

      </Box>
    </ThemeProvider>
  );
}

export default App;
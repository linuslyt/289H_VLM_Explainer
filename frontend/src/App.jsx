import {
  Box,
  CssBaseline,
  Divider,
  ThemeProvider,
  Typography,
  createTheme,
  CircularProgress
} from '@mui/material';
import { useCallback, useState } from 'react';
import { mockExplanationData } from './mockData'; // Import mock data
import ImageUploader from './components/ImageUploader';
import TokenSelector from './components/TokenSelector';
import LogConsole from './components/LogConsole';
import Visualizations from './components/Visualizations';

// Create a dark/clean theme or keep default. 
// Using default here but removing body margins via CssBaseline.
const theme = createTheme();

function App() {
  // --- STATE ---
  const [logs, setLogs] = useState([]);
  const [preview, setPreview] = useState(null);
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

  // Handle Token Click (using mock data)
  const handleTokenClick = (token) => {
      setSelectedToken(token);
      setIsExplaining(true);
      setExplanationData(null);
      
      // Simulate API call with a delay
      setTimeout(() => {
          setExplanationData(mockExplanationData);
          addLog("Backend: Mock explanation received.");
          setIsExplaining(false);
      }, 500); // Simulate network delay
  };

  // Image drag and drop handler (using mock data)
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

    // mock backend logs
    addLog("Network: Uploading image to server...");
    
    setTimeout(() => addLog("Backend: Image received. Resizing..."), 800);
    setTimeout(() => addLog("Backend: Loading inference model (ViT-GPT2)..."), 1800);
    setTimeout(() => addLog("Backend: Generating attention masks..."), 3000);
    setTimeout(()=> {
      addLog("Backend: Success. Caption generated.");
      setCaption("a large brown dog running through green grass");
        setIsProcessing(false);
    }, 1000); // Simulate network delay

  }, []);

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
          width: '50%', 
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
          explanationData={explanationData}
        />

      </Box>
    </ThemeProvider>
  );
}

export default App;
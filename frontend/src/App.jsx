import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import TerminalIcon from '@mui/icons-material/Terminal';
import {
  Box,
  CssBaseline,
  Divider,
  ThemeProvider,
  Typography,
  createTheme,
  CircularProgress
} from '@mui/material';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import ConceptCard from './components/ConceptCard';
import ConceptChart from './components/ConceptChart';
import { mockExplanationData } from './mockData'; // Import mock data

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

  const consoleEndRef = useRef(null);

  // Logging helper
  const addLog = (message) => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    setLogs((prev) => [...prev, `[${time}] ${message}`]);
  };

  // Auto-scroll logs only 
  const logsContainerRef = useRef(null); // Ref will be assigned to console container
  const isUserAtBottom = useRef(true);
  const handleScroll = () => {
    const { scrollTop, scrollHeight, clientHeight } = logsContainerRef.current;
    
    // Calculate distance from bottom
    const distanceToBottom = scrollHeight - (scrollTop + clientHeight);
    
    // If we are within 50px of the bottom, we are "at the bottom"
    // Otherwise, the user has scrolled up.
    isUserAtBottom.current = distanceToBottom < 50;
  };

  useEffect(() => {
    // Only scroll if the user was already at the bottom
    if (isUserAtBottom.current && logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }
  }, [logs]);

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

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop, 
    accept: {'image/*': []},
    multiple: false
  });

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
          <Box sx={{ 
            height: '50%', 
            p: 2, 
            display: 'flex', 
            flexDirection: 'column' 
          }}>
            <Typography variant="overline" color="text.secondary" fontWeight="bold">
              Input Source
            </Typography>
            
            <Box 
              {...getRootProps()} 
              sx={{
                flex: 1,
                border: '2px dashed',
                borderColor: isDragActive ? 'primary.main' : '#e0e0e0',
                borderRadius: 2,
                bgcolor: isDragActive ? 'action.hover' : '#fafafa',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                overflow: 'hidden',
                position: 'relative',
                transition: 'all 0.2s ease'
              }}
            >
              <input {...getInputProps()} />
              {preview ? (
                <img 
                  src={preview} 
                  alt="Upload preview" 
                  style={{ width: '100%', height: '100%', objectFit: 'contain' }} 
                />
              ) : (
                <Box textAlign="center" color="text.secondary">
                  <CloudUploadIcon sx={{ fontSize: 48, mb: 1, color: '#bdbdbd' }} />
                  <Typography variant="body1">
                    {isDragActive ? "Drop image here..." : "Drag & drop or Click to Upload"}
                  </Typography>
                </Box>
              )}
            </Box>
          </Box>

          <Divider />

          {/* Caption display, token selector */}
          <Box sx={{ 
            height: '15%', // Adjusted slightly to ensure text fits comfortably
            p: 2, 
            display: 'flex', 
            flexDirection: 'column',
            bgcolor: '#fff'
          }}>
             <Box display="flex" justifyContent="space-between" alignItems="center">
                <Typography variant="overline" color="text.secondary" fontWeight="bold">
                  Generated Caption
                </Typography>
                {selectedToken && (
                   <Typography variant="caption" sx={{ bgcolor: 'primary.main', color: 'white', px: 1, borderRadius: 1 }}>
                     Token to analyze: {selectedToken}
                   </Typography>
                )}
             </Box>

             <Box sx={{ 
               flex: 1, 
               display: 'flex', 
               alignItems: 'center', 
               justifyContent: 'center',
               flexWrap: 'wrap',
               gap: 1
             }}>
                {isProcessing ? (
                    <CircularProgress size={24} />
                ) : !caption ? (
                  <Typography variant="body2" color="text.disabled" fontStyle="italic">
                    Upload image for captioning to begin...
                  </Typography>
                ) : (
                  caption.split(' ').map((word, i) => (
                    <Typography
                      key={i}
                      onClick={() => handleTokenClick(word)}
                      sx={{
                        cursor: 'pointer',
                        px: 0.8,
                        py: 0.4,
                        borderRadius: 1,
                        fontSize: '1.1rem',
                        border: selectedToken === word ? '1px solid' : '1px solid transparent',
                        borderColor: 'primary.main',
                        bgcolor: selectedToken === word ? 'primary.light' : 'transparent',
                        color: selectedToken === word ? 'white' : 'text.primary',
                        transition: 'all 0.1s',
                        '&:hover': { bgcolor: 'grey.200', color: 'black' }
                      }}
                    >
                      {word}
                    </Typography>
                  ))
                )}
             </Box>
          </Box>

          <Divider />

          {/* Streamed logs display */}
          <Box sx={{ 
            flex: 1,
            bgcolor: '#1e1e1e', 
            color: '#00ff41',
            p: 2,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden'
          }}>
            <Box display="flex" alignItems="center" gap={1} mb={1} sx={{ opacity: 0.7 }}>
               <TerminalIcon fontSize="small" sx={{ color: '#fff' }}/>
               <Typography variant="caption" sx={{ color: '#fff', textTransform: 'uppercase' }}>
                 Backend Stream
               </Typography>
            </Box>
            <Box 
              ref={logsContainerRef}
              onScroll={handleScroll}
              sx={{ 
                flex: 1, 
                overflowY: 'auto',
                fontFamily: 'monospace',
                fontSize: '0.85rem'
              }}
            >
              {logs.length === 0 && (
                <Typography variant="body2" sx={{ opacity: 0.3 }}>Ready...</Typography>
              )}
              {logs.map((log, i) => (
                <div key={i} style={{ marginBottom: '4px', wordBreak: 'break-all' }}>{log}</div>
              ))}
            </Box>
          </Box>
        </Box>

        {/* Concept visualizer (Right Panel) */}
        <Box sx={{ 
          width: '50%', 
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          bgcolor: '#fafafa',
          borderLeft: '1px solid #e0e0e0',
          position: 'relative'
        }}>
           {!selectedToken ? (
             <Box display="flex" flex={1} alignItems="center" justifyContent="center" flexDirection="column" color="text.secondary">
                <Typography variant="h6">Select a token to analyze</Typography>
                <Typography variant="body2">Click on a word in the caption on the left</Typography>
             </Box>
           ) : (
             <Box display="flex" flexDirection="column" height="100%">
               {/* Header for Right Panel */}
               <Box p={2} borderBottom="1px solid #e0e0e0" bgcolor="white">
                 <Typography variant="h6">
                   Explanation for: <Box component="span" color="primary.main" fontWeight="bold">"{selectedToken}"</Box>
                 </Typography>
               </Box>

               {/* Comparison Columns */}
               {isExplaining || !explanationData ? (
                   <Box flex={1} display="flex" alignItems="center" justifyContent="center">
                       <CircularProgress />
                   </Box>
               ) : (
               <Box display="flex" flex={1} overflow="hidden">
                 {/* Global Activations */}
                 <Box flex={1} display="flex" flexDirection="column" borderRight="1px solid #e0e0e0">
                    <Box p={2} bgcolor="#e3f2fd" borderBottom="1px solid #bbdefb">
                      <Typography variant="subtitle1" fontWeight="bold" color="#1565c0">
                        Global Activation (CoX-LMM)
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        What is present in the internal state?
                      </Typography>
                    </Box>
                    {/* Chart for Global */}
                    <Box px={2} pt={2} borderBottom="1px solid #eee" bgcolor="#f1f8ff">
                        <ConceptChart data={explanationData.global_activations} color="#2196f3" />
                    </Box>
                    <Box flex={1} p={2} sx={{ overflowY: 'auto' }}>
                      {explanationData.global_activations.map((concept, idx) => (
                        <ConceptCard key={idx} concept={concept} type="global" />
                      ))}
                    </Box>
                 </Box>

                 {/* Local Importance */}
                 <Box flex={1} display="flex" flexDirection="column">
                    <Box p={2} bgcolor="#e8f5e9" borderBottom="1px solid #c8e6c9">
                      <Typography variant="subtitle1" fontWeight="bold" color="#2e7d32">
                        Local Importance (Gradient)
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        What actually caused this prediction?
                      </Typography>
                    </Box>
                    {/* Chart for Local */}
                    <Box px={2} pt={2} borderBottom="1px solid #eee" bgcolor="#f1f8ff">
                        <ConceptChart data={explanationData.local_importances} color="#4caf50" />
                    </Box>
                    <Box flex={1} p={2} sx={{ overflowY: 'auto' }}>
                      {explanationData.local_importances.map((concept, idx) => (
                        <ConceptCard key={idx} concept={concept} type="local" />
                      ))}
                    </Box>
                 </Box>
               </Box>
               )}
             </Box>
           )}
        </Box>

      </Box>
    </ThemeProvider>
  );
}

export default App;

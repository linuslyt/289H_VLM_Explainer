import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import TerminalIcon from '@mui/icons-material/Terminal';
import {
  Box,
  CssBaseline,
  Divider,
  ThemeProvider,
  Typography,
  createTheme
} from '@mui/material';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useDropzone } from 'react-dropzone';

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

  // Image drag and drop handler
  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // show image preview
    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);

    // reset state
    setLogs([]);
    setCaption(null);
    setSelectedToken(null);
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
    }, 4500);

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
                {!caption ? (
                  <Typography variant="body2" color="text.disabled" fontStyle="italic">
                    Upload image for captioning to begin...
                  </Typography>
                ) : (
                  caption.split(' ').map((word, i) => (
                    <Typography
                      key={i}
                      onClick={() => setSelectedToken(word)}
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

        {/* Concept visualizer */}
        <Box sx={{ 
          width: '50%', 
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: '#fafafa',
          borderLeft: '1px solid #e0e0e0',
          position: 'relative'
        }}>
          <Typography variant="h5" color="text.disabled" fontWeight="bold">
            Concept Visualizations Area
          </Typography>
          <Box>
            {/* TODO: d3 visualizations here */}
          </Box>
        </Box>

      </Box>
    </ThemeProvider>
  );
}

export default App;
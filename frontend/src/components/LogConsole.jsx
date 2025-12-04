import React, { useRef, useEffect } from 'react';
import { Box, Typography } from '@mui/material';
import TerminalIcon from '@mui/icons-material/Terminal';

const LogConsole = ({ logs }) => {
  const logsContainerRef = useRef(null);
  const isUserAtBottom = useRef(true);

  const handleScroll = () => {
    const { scrollTop, scrollHeight, clientHeight } = logsContainerRef.current;
    const distanceToBottom = scrollHeight - (scrollTop + clientHeight);
    isUserAtBottom.current = distanceToBottom < 50;
  };

  useEffect(() => {
    if (isUserAtBottom.current && logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }
  }, [logs]);

  return (
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
  );
};

export default LogConsole;

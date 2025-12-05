import TerminalIcon from '@mui/icons-material/Terminal';
import { Box, Typography } from '@mui/material';
import { useLayoutEffect, useRef } from 'react';

const LogConsole = ({ logs }) => {
  // we use refs instead of useState to prevent rerenders
  const logsContainerRef = useRef(null);
  const isAtBottomRef = useRef(true); // autoscroll if at bottom

  const handleScroll = () => {
    if (!logsContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = logsContainerRef.current;
    const distanceToBottom = scrollHeight - (scrollTop + clientHeight);

    // high pixel tolerance to prevent sub-pixel math errors from breaking autoscroll
    const isClose = distanceToBottom <= 50;

    // only update if the status changes to avoid unnecessary noise
    if (isAtBottomRef.current !== isClose) {
      isAtBottomRef.current = isClose;
    }
  };

  // useLayoutEffect runs synchronously after DOM mutations but before paint
  // maintains autoscroll behavior even with rapid log ingest
  useLayoutEffect(() => {
    if (isAtBottomRef.current && logsContainerRef.current) {
      const node = logsContainerRef.current;
      
      // scroll to bottom
      node.scrollTop = node.scrollHeight;
      
      // manually set flag to maintain the autoscroll for the next log in the batch
      isAtBottomRef.current = true;
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
           Logs
         </Typography>
      </Box>
      
      <Box 
        ref={logsContainerRef}
        onScroll={handleScroll}
        sx={{ 
          flex: 1, 
          overflowY: 'auto',
          fontFamily: 'monospace',
          fontSize: '0.85rem',
          scrollBehavior: 'smooth', 

          scrollbarWidth: 'thin',
          scrollbarColor: 'rgba(255, 255, 255, 0.3) transparent',
          '&::-webkit-scrollbar': { width: '6px' },
          '&::-webkit-scrollbar-track': { background: 'transparent' },
          '&::-webkit-scrollbar-thumb': { 
            backgroundColor: 'rgba(255, 255, 255, 0.3)', 
            borderRadius: '3px' 
          },
          '&::-webkit-scrollbar-thumb:hover': { backgroundColor: '#ffffff' }
        }}
      >
        {logs.length === 0 && (
          <Typography variant="body2" sx={{ opacity: 0.3 }}>Ready...</Typography>
        )}
        
        {logs.map((log, i) => (
          <div key={i} style={{ marginBottom: '4px', wordBreak: 'break-all' }}>
            {log}
          </div>
        ))}
      </Box>
    </Box>
  );
};

export default LogConsole;
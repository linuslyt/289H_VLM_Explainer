import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { Box, Typography } from '@mui/material';
import { useDropzone } from 'react-dropzone';

const ImageUploader = ({ onDrop, preview }) => {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop, 
    accept: {'image/*': []},
    multiple: false
  });

  return (
    <Box sx={{ 
      height: '40%', 
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
  );
};

export default ImageUploader;

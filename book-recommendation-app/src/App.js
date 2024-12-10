import React, { useState } from 'react';
import { 
  Button, Typography, Card, CardContent, Grid, Box, 
  LinearProgress, Container, Paper
} from '@mui/material';
import BookIcon from '@mui/icons-material/Book';

function App() {
  const [recommendation, setRecommendation] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleRecommendation = async () => {
    setLoading(true);
    // 여기에 실제 API 호출 로직을 구현합니다.
    // 예시를 위해 setTimeout을 사용합니다.
    setTimeout(() => {
      setRecommendation({
        title: "The Great Gatsby",
        description: "A novel by F. Scott Fitzgerald set in the Jazz Age on Long Island.",
        ndcg: 0.85,
        hitRatio: 0.72
      });
      setLoading(false);
    }, 2000);
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          Book Recommendation System
        </Typography>
        <Button 
          variant="contained" 
          color="primary" 
          onClick={handleRecommendation}
          startIcon={<BookIcon />}
          disabled={loading}
        >
          Get Recommendation
        </Button>

        {loading && <LinearProgress sx={{ my: 2 }} />}

        {recommendation && (
          <Paper elevation={3} sx={{ mt: 3, p: 2 }}>
            <Typography variant="h5" component="h2" gutterBottom>
              Recommended Book
            </Typography>
            <Typography variant="h6" color="primary" gutterBottom>
              {recommendation.title}
            </Typography>
            <Card variant="outlined" sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="body1">
                  {recommendation.description}
                </Typography>
              </CardContent>
            </Card>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="subtitle1">NDCG Score</Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={recommendation.ndcg * 100} 
                  sx={{ height: 10, borderRadius: 5 }}
                />
                <Typography variant="body2" align="right">
                  {(recommendation.ndcg * 100).toFixed(2)}%
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="subtitle1">Hit Ratio</Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={recommendation.hitRatio * 100} 
                  sx={{ height: 10, borderRadius: 5 }}
                />
                <Typography variant="body2" align="right">
                  {(recommendation.hitRatio * 100).toFixed(2)}%
                </Typography>
              </Grid>
            </Grid>
          </Paper>
        )}
      </Box>
    </Container>
  );
}

export default App;

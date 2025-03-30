import { useContext } from 'react';
import { AppContext } from '../context/AppContext.jsx';

export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
} 
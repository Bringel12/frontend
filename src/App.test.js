import { render, screen } from '@testing-library/react';
import App from './App';

test('renderiza o título principal do app', () => {
  render(<App />);
  const heading = screen.getByText(/Assistente RAG com busca em SQL/i);
  expect(heading).toBeInTheDocument();
});

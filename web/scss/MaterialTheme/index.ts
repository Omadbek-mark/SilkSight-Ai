import { createTheme } from '@mui/material/styles';

// Lapis blue and saffron from Timurid tilework; type pairing follows Hotelia.
export const silkSightTheme = createTheme({
	palette: {
		primary: { main: '#1d3f6e', contrastText: '#fff' },
		secondary: { main: '#b8622a', contrastText: '#fff' },
		background: { default: '#faf7f1', paper: '#fff' },
		text: { primary: '#1c1c19', secondary: '#5f6670' },
		divider: '#e6e1d8',
	},
	shape: { borderRadius: 8 },
	typography: {
		fontFamily: '"Hanken Grotesk", Arial, sans-serif',
		h1: { fontFamily: '"Newsreader", Georgia, serif', fontWeight: 400 },
		h2: { fontFamily: '"Newsreader", Georgia, serif', fontWeight: 400 },
		h3: { fontFamily: '"Newsreader", Georgia, serif', fontWeight: 400 },
		h4: { fontFamily: '"Newsreader", Georgia, serif', fontWeight: 500 },
		button: { textTransform: 'none', fontWeight: 600 },
	},
	components: {
		MuiCssBaseline: { styleOverrides: { body: { backgroundColor: '#faf7f1' } } },
		MuiButton: {
			defaultProps: { disableElevation: true },
			styleOverrides: { root: { minHeight: 44, padding: '10px 20px' } },
		},
	},
});

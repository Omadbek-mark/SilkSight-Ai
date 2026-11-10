import { CssBaseline } from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import { appWithTranslation } from 'next-i18next';
import type { AppProps } from 'next/app';
import 'leaflet/dist/leaflet.css';
import '../scss/app.scss';
import { silkSightTheme } from '../scss/MaterialTheme';

const App = ({ Component, pageProps }: AppProps) => (
	<ThemeProvider theme={silkSightTheme}>
		<CssBaseline />
		<Component {...pageProps} />
	</ThemeProvider>
);

export default appWithTranslation(App, require('../next-i18next.config'));

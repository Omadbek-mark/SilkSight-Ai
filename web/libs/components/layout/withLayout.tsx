import { ComponentType } from 'react';
import PublicLayout from './PublicLayout';

export default function withLayout<P extends object>(Component: ComponentType<P>) {
	return function SilkSightPage(props: P) {
		return (
			<PublicLayout>
				<Component {...props} />
			</PublicLayout>
		);
	};
}

#include "ViewerInterface.h"

int init(int argc, char* argv[]) {
	QApplication app(argc, argv);

	QPixmap splashImage("./sourcefiles/data/splash.png");
	QSplashScreen splash(splashImage);
	splash.show();
	app.processEvents();

	QIcon icon;
	icon.addFile("./data/icon_16.png");
	icon.addFile("./data/icon_32.png");
	icon.addFile("./data/icon_48.png");
	icon.addFile("./data/icon_64.png");
	icon.addFile("./data/icon_96.png");
	icon.addFile("./data/icon_128.png");
	icon.addFile("./data/icon_192.png");
	icon.addFile("./data/icon_256.png");
	app.setWindowIcon(icon);

	QString openWithFilename;
	if (QCoreApplication::arguments().size() > 1) {
		openWithFilename = QCoreApplication::arguments().at(1);
	}

	ct::ViewerInterface* mainInterface = new ct::ViewerInterface(openWithFilename);
	mainInterface->show();
	splash.finish(mainInterface);

	//QMainWindow window;
	//window.setCentralWidget(mainInterface);
	//window.show();

	return app.exec();
}

int main(int argc, char* argv[]) {
	return init(argc, argv);
}
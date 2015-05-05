#ifndef CT_MAININTERFACE
#define CT_MAININTERFACE

//Qt
#include <QtWidgets/QtWidgets>

namespace ct {

	class MainInterface : public QWidget{
		Q_OBJECT
	public:
		MainInterface(QWidget *parent = 0);
		~MainInterface();
		QSize sizeHint() const;
	private:
		QHBoxLayout* _mainLayout;
		QVBoxLayout* _leftLayout;
		QHBoxLayout* _openLayout;
		QLineEdit* _inputFileEdit;
		QPushButton* _browseButton;
		QPushButton* _loadButton;
		QProgressBar* _progressBar;
	};

}

#endif
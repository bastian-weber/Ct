#ifndef CT_MAININTERFACE
#define CT_MAININTERFACE

#include <future>

//Qt
#include <QtWidgets/QtWidgets>

#include "ImageView.h"
#include "CtVolume.h"

namespace ct {

	class MainInterface : public QWidget{
		Q_OBJECT
	public:
		MainInterface(QWidget *parent = 0);
		~MainInterface();
		QSize sizeHint() const;
	private:
		CtVolume _volume;

		QVBoxLayout* _mainLayout;
		QHBoxLayout* _subLayout;
		QVBoxLayout* _leftLayout;
		QHBoxLayout* _openLayout;
		QLabel* _openLabel;
		QLineEdit* _inputFileEdit;
		QPushButton* _browseButton;
		QPushButton* _loadButton;
		QPushButton* _reconstructButton;
		QPushButton* _saveButton;
		QProgressBar* _progressBar;
		hb::ImageView* _imageView;
	private slots:
		void reactToBrowseButtonClick();
		void reactToLoadButtonClick();
		void reactToReconstructButtonClick();
		void reactToLoadProgressUpdate();
		void reactToLoadCompletion();
		void reactToReconstructionProgressUpdate();
		void reactToReconstructionCompletion();
		void reactToSaveProgressUpdate();
		void reactToSaveCompletion();
	};

}

#endif
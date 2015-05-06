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
		void disableAllControls();
		void startupState();
		void fileSelectedState();
		void preprocessedState();
		void reconstructedState();

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
		void reactToTextChange(QString text);
		void reactToBrowseButtonClick();
		void reactToLoadButtonClick();
		void reactToReconstructButtonClick();
		void reactToSaveButtonClick();
		void reactToLoadProgressUpdate(double percentage);
		void reactToLoadCompletion(CtVolume::LoadStatus status);
		void reactToReconstructionProgressUpdate(double percentage);
		void reactToReconstructionCompletion(CtVolume::ReconstructStatus status);
		void reactToSaveProgressUpdate(double percentage);
		void reactToSaveCompletion(CtVolume::SaveStatus status);
	};

}

#endif
#ifndef CT_IMPORTSETTINGSDIALOG
#define CT_IMPORTSETTINGSDIALOG

#include <iostream>
#include <memory>
#include <vector>

//Qt
#include <QtCore/QtCore>
#include <QtGui/QtGui>
#include <QtWidgets/QtWidgets>

namespace ct {

	class ImportSettingsDialog : public QDialog {
		Q_OBJECT
	public:
		ImportSettingsDialog(std::shared_ptr<QSettings> settings, QWidget* parent = 0);
		~ImportSettingsDialog();
		int execForFilesize(size_t requiredSize);
		size_t getXSize() const;
		size_t getYSize() const;
		size_t getZSize() const;
	protected:
		void showEvent(QShowEvent* e);
	private:
		//functions

		//variables
		std::shared_ptr<QSettings> settings;
		size_t requiredSize = 0;
		//widgets
		QVBoxLayout* mainLayout;
		QFormLayout* formLayout;
		QHBoxLayout* buttonLayout;
		QPushButton* okButton;
		QPushButton* cancelButton;
		QSpinBox* xSpinBox;
		QSpinBox* ySpinBox;
		QSpinBox* zSpinBox;
		QLabel* actualSizeLabel;
		QLabel* requiredSizeLabel;
	private slots:
		void updateSize();
	};
}
#endif
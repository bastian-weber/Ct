#ifndef CT_CUDASETTINGSDIALOG
#define CT_CUDASETTINGSDIALOG

#include <iostream>
#include <memory>
#include <vector>

//Qt
#include <QtCore/QtCore>
#include <QtGui/QtGui>
#include <QtWidgets/QtWidgets>

namespace ct {

	class CudaSettingsDialog : public QDialog {
		Q_OBJECT
	public:
		CudaSettingsDialog(std::shared_ptr<QSettings> settings, std::vector<std::string> const& devices, QWidget* parent = 0);
		~CudaSettingsDialog();
		std::vector<int> getActiveCudaDevices() const;
	protected:
		void showEvent(QShowEvent* e);
	private:
		//functions

		//variables
		std::shared_ptr<QSettings> settings;
		//widgets
		QVBoxLayout* mainLayout;
		std::vector<QCheckBox*> checkboxes;
		QHBoxLayout* buttonLayout;
		QPushButton* okButton;
		QPushButton* cancelButton;
	private slots:
		void reactToOkButtonClick();
		void reactToCheckboxToggle();
	signals:
		void dialogConfirmed();
		void dialogClosed();
	};
}
#endif
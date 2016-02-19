#include "CudaSettingsDialog.h"

namespace ct {

	CudaSettingsDialog::CudaSettingsDialog(std::shared_ptr<QSettings> settings, std::vector<std::string> const& devices, QWidget* parent)
		: settings(settings), 
		checkboxes(devices.size()),
		QDialog(parent) {
		this->setWindowModality(Qt::WindowModal);
		this->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
		this->setWindowFlags(this->windowFlags() & ~Qt::WindowContextHelpButtonHint);

		this->setWindowTitle(tr("Cuda Settings"));

		this->okButton = new QPushButton(tr("&Ok"), this);
		this->okButton->setDefault(true);
		QObject::connect(this->okButton, SIGNAL(clicked()), this, SLOT(reactToOkButtonClick()));
		QObject::connect(this->okButton, SIGNAL(clicked()), this, SIGNAL(dialogClosed()));

		this->cancelButton = new QPushButton(tr("&Cancel"), this);
		QObject::connect(this->cancelButton, SIGNAL(clicked()), this, SLOT(close()));
		QObject::connect(this->cancelButton, SIGNAL(clicked()), this, SIGNAL(dialogClosed()));

		this->buttonLayout = new QHBoxLayout();
		this->buttonLayout->addStretch(1);
		this->buttonLayout->addWidget(okButton);
		this->buttonLayout->addWidget(cancelButton);

		this->mainLayout = new QVBoxLayout();

		//generate the checkboxes for all the devices
		for (int i = 0; i < devices.size(); ++i) {
			this->checkboxes[i] = new QCheckBox(devices[i].c_str());
			QObject::connect(this->checkboxes[i], SIGNAL(stateChanged(int)), this, SLOT(reactToCheckboxToggle()));
			mainLayout->addWidget(this->checkboxes[i]);
		}

		this->mainLayout->addLayout(this->buttonLayout);

		this->setLayout(this->mainLayout);
		this->layout()->setSizeConstraint(QLayout::SetFixedSize);
	}

	CudaSettingsDialog::~CudaSettingsDialog() {
		delete this->mainLayout;
		delete this->buttonLayout;
		delete this->okButton;
		delete this->cancelButton;
		for (int i = 0; i < this->checkboxes.size(); ++i) {
			delete this->checkboxes[i];
		}
	}

	std::vector<int> CudaSettingsDialog::getActiveCudaDevices() const {
		//load settings
		std::vector<int> devices;
		QVariantList standard;
		standard << 0;
		QList<QVariant> deviceIds = this->settings->value("activeCudaDevices", standard).toList();
		if (deviceIds.size() == 0) deviceIds = standard;
		for (QList<QVariant>::const_iterator i = deviceIds.begin(); i != deviceIds.end(); ++i) {
			if (i->toInt() < this->checkboxes.size()) {
				devices.push_back(i->toInt());
			}
		}
		return devices;
	}

	void CudaSettingsDialog::showEvent(QShowEvent * e) {
		std::vector<int> activeDevices = this->getActiveCudaDevices();
		for (int i = 0; i < activeDevices.size(); ++i) {
			if (activeDevices[i] < this->checkboxes.size()) {
				this->checkboxes[activeDevices[i]]->setChecked(true);
			}
		}
	}

	void CudaSettingsDialog::reactToCheckboxToggle() {
		int checkedCnt = 0;
		for (int i = 0; i < this->checkboxes.size(); ++i) {
			if (this->checkboxes[i]->isChecked()) ++checkedCnt;
		}
		if (checkedCnt == 0) {
			this->okButton->setEnabled(false);
		} else {
			this->okButton->setEnabled(true);
		}
	}

	//============================================================================== PROTECTED ==============================================================================\\


	//=============================================================================== PRIVATE ===============================================================================\\


	//============================================================================ PRIVATE SLOTS =============================================================================\\

	void CudaSettingsDialog::reactToOkButtonClick() {
		QVariantList deviceIds;
		for (int i = 0; i < this->checkboxes.size(); ++i) {
			if (this->checkboxes[i]->isChecked()) deviceIds << i;
		}
		settings->setValue("activeCudaDevices", deviceIds);
		emit(dialogConfirmed());
		this->close();
	}

}
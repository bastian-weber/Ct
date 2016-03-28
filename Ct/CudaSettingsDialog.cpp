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

		this->devicesLayout = new QVBoxLayout();

		//generate the checkboxes for all the devices
		for (int i = 0; i < devices.size(); ++i) {
			this->checkboxes[i] = new QCheckBox(devices[i].c_str(), this);
			QObject::connect(this->checkboxes[i], SIGNAL(stateChanged(int)), this, SLOT(reactToCheckboxToggle()));
			devicesLayout->addWidget(this->checkboxes[i]);
		}
		this->devicesGroupBox = new QGroupBox(tr("CUDA Devices"), this);
		this->devicesGroupBox->setLayout(devicesLayout);

		this->memorySpinBox = new QSpinBox(this);
		this->memorySpinBox->setMinimum(0);
		this->memorySpinBox->setMaximum(50000);
		this->memorySpinBox->setSuffix(tr("Mb"));
		this->memorySpinBox->setSingleStep(1);
		this->memoryLayout = new QVBoxLayout;
		this->memoryLayout->addWidget(this->memorySpinBox, 0, Qt::AlignLeft);
		this->memoryGroupBox = new QGroupBox(tr("Amount of GPU memory to spare"), this);
		this->memoryGroupBox->setLayout(this->memoryLayout);

		this->mainLayout = new QVBoxLayout;
		this->mainLayout->addWidget(devicesGroupBox);
		this->mainLayout->addWidget(memoryGroupBox);
		this->mainLayout->addLayout(this->buttonLayout);

		this->setLayout(this->mainLayout);
		this->layout()->setSizeConstraint(QLayout::SetFixedSize);

		//sets the default values
		this->showEvent(&QShowEvent());
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

	int CudaSettingsDialog::getSpareMemoryAmount() const {
		return this->memorySpinBox->value();
	}

	void CudaSettingsDialog::showEvent(QShowEvent * e) {
		this->setDefaultValues();
	}

	void CudaSettingsDialog::setDefaultValues() {
		std::vector<int> activeDevices = this->getActiveCudaDevices();
		for (int i = 0; i < activeDevices.size(); ++i) {
			if (activeDevices[i] < this->checkboxes.size()) {
				this->checkboxes[activeDevices[i]]->setChecked(true);
			}
		}

		this->memorySpinBox->setValue(this->settings->value("gpuSpareMemory", 0).toLongLong());
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
		settings->setValue("gpuSpareMemory", this->memorySpinBox->value());
		emit(dialogConfirmed());
		this->close();
	}

}
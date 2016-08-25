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
		this->memorySpinBox->setMaximum(100000);
		this->memorySpinBox->setSuffix(tr("Mb"));
		this->memorySpinBox->setSingleStep(1);
		this->memoryLayout = new QVBoxLayout;
		this->memoryLayout->addWidget(this->memorySpinBox, 0, Qt::AlignLeft);
		this->memoryGroupBox = new QGroupBox(tr("Amount of GPU memory to spare"), this);
		this->memoryGroupBox->setLayout(this->memoryLayout);

		this->memoryCoefficientSpinBox = new QDoubleSpinBox(this);
		this->memoryCoefficientSpinBox->setValue(1);
		this->memoryCoefficientSpinBox->setSingleStep(0.1);
		this->memoryCoefficientSpinBox->setDecimals(3);
		this->memoryCoefficientSpinBox->setRange(-100000, 100000);
		this->multiprocessorSpinBox = new QDoubleSpinBox(this);
		this->multiprocessorSpinBox->setValue(1);
		this->multiprocessorSpinBox->setSingleStep(0.1);
		this->multiprocessorSpinBox->setDecimals(3);
		this->multiprocessorSpinBox->setRange(-100000, 100000);
		this->coefficientsLayout = new QFormLayout;
		this->coefficientsLayout->addRow(tr("Multiprocessor coefficient:"), this->multiprocessorSpinBox);
		this->coefficientsLayout->addRow(tr("Memory bandwidth coefficient:"), this->memoryCoefficientSpinBox);
		this->coefficientsGroupBox = new QGroupBox(tr("Multi-GPU coefficients"), this);
		this->coefficientsGroupBox->setLayout(this->coefficientsLayout);

		this->gpuPreprocessingCheckbox = new QCheckBox("Use GPU Preprocessing", this);
		this->preprocessingLayout = new QVBoxLayout;
		this->preprocessingLayout->addWidget(this->gpuPreprocessingCheckbox);
		this->preprocessingGroupBox = new QGroupBox(tr("Preprocessing"), this);
		this->preprocessingGroupBox->setLayout(this->preprocessingLayout);

		this->mainLayout = new QVBoxLayout;
		this->mainLayout->addWidget(this->devicesGroupBox);
		this->mainLayout->addWidget(this->memoryGroupBox);
		this->mainLayout->addWidget(this->coefficientsGroupBox);
		this->mainLayout->addWidget(this->preprocessingGroupBox);
		this->mainLayout->addLayout(this->buttonLayout);

		this->setLayout(this->mainLayout);
		this->layout()->setSizeConstraint(QLayout::SetFixedSize);

		this->setDefaultValues();
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

	double CudaSettingsDialog::getMemoryBandwidthCoefficient() const {
		return this->memoryCoefficientSpinBox->value();
	}

	double CudaSettingsDialog::getMultiprocessorCoefficient() const {
		return this->multiprocessorSpinBox->value();
	}

	bool CudaSettingsDialog::getUseGpuPreprocessing() const {
		return this->gpuPreprocessingCheckbox->isChecked();
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
		this->multiprocessorSpinBox->setValue(this->settings->value("gpuMultiprocessorCoefficient", 1).toDouble());
		this->memoryCoefficientSpinBox->setValue(this->settings->value("gpuMemoryBandwidthCoefficient", 1).toDouble());
		this->gpuPreprocessingCheckbox->setChecked(this->settings->value("useGpuPreprocessing", true).toBool());
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
		settings->setValue("gpuMultiprocessorCoefficient", this->multiprocessorSpinBox->value());
		settings->setValue("gpuMemoryBandwidthCoefficient", this->memoryCoefficientSpinBox->value());
		settings->setValue("useGpuPreprocessing", this->gpuPreprocessingCheckbox->isChecked());
		emit(dialogConfirmed());
		this->close();
	}

}
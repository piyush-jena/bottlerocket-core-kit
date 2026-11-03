Name: %{_cross_os}nvme-cli
Version: 3.2
Release: 1%{?dist}
Epoch: 1
Summary: CLI to interact with NVMe devices
License: GPL-2.0-only AND LGPL-2.1-or-later AND CC0-1.0 AND MIT AND LicenseRef-Gary-S-Brown-CRC32
URL: https://github.com/linux-nvme/nvme-cli
Source0: https://github.com/linux-nvme/nvme-cli/archive/v%{version}/nvme-cli-%{version}.tar.gz

# nvme-cli 3.x regression: the bundled libnvme rejects device paths given as
# udev symlinks, such as /dev/disk/by-ebs-id/* or /dev/disk/by-id/*. Drop this
# once upstream libnvme resolves symlinks itself.
Patch1001: 1001-libnvme-resolve-device-symlinks.patch

BuildRequires: meson
BuildRequires: %{_cross_os}glibc-devel
BuildRequires: %{_cross_os}libjson-c-devel
Requires: %{_cross_os}libjson-c

%description
%{summary}.

%package devel
Summary: Files for development using the library for NVM Express
Requires: %{name}

%description devel
%{summary}.

%prep
%autosetup -n nvme-cli-%{version} -p1

%build
CONFIGURE_OPTS=(
 -Ddocs=false
 -Djson-c=enabled
 -Dpython=disabled
 -Dopenssl=disabled
 -Dkeyutils=disabled
 -Dliburing=disabled
 -Dlibdbus=disabled
 -Dlibkmod=disabled
 -Dlibarchive=disabled

 # Keep the NVMe-oF, NVMe-MI and SED (TCG Opal) commands, and build the
 # nvme top dashboard
 -Dfabrics=enabled
 -Dmi=enabled
 -Dsed=enabled
 -Dtop=enabled

 # Keep the 2.x command names, ghostdog runs `nvme id-ctrl`
 -Ddeprecated-cmds=enabled

 # Don't build the technology preview privilege separation helper and
 # decoder sandbox
 -Dprivsep=disabled
 -Dsandbox=disabled

 # Don't build the NVMe-oF discovery and key daemons nor install the udev,
 # systemd, dracut and NetworkManager autoconnect integration
 -Dnvme-discoverd=disabled
 -Dmdns=disabled
 -Dnvme-keysd=disabled
 -Dkmip=disabled
 -Dnvmf-autoconnect=disabled

 # Build the bundled libnvme3, which nvme uses and this package ships
 -Dlibnvme=enabled
)

%cross_meson "${CONFIGURE_OPTS[@]}"
%cross_meson_build

%install
%cross_meson_install
# This is a reference-only example of the NVMe-oF connection configuration,
# nvme-cli doesn't read it
rm %{buildroot}%{_sysconfdir}/nvme/nvme-fabrics.conf.sample

%files
%license LICENSE libnvme/COPYING LICENSES/LGPL-2.1-or-later.txt LICENSES/MIT.txt LICENSES/CC0-1.0.txt LICENSES/LicenseRef-Gary-S-Brown-CRC32.txt
%{_cross_attribution_file}
%{_cross_sbindir}/nvme
%{_cross_libdir}/libnvme3.so.*
%exclude %{_cross_udevrulesdir}
%exclude %{_cross_datadir}

%files devel
%{_cross_includedir}/libnvme3
%{_cross_libdir}/libnvme3.so
%{_cross_pkgconfigdir}/libnvme3.pc

%changelog

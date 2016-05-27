PWs=   wimaging.rstw aa.rstw wkernel.rstw
RSTs=  $(PWs:.rstw=.rst)
HTMLs= $(PWs:.rstw=.html)

ifeq ($(OS),Windows_NT)
	PWEAVE=pweave.bat
	PPATH=..\;${PYTHONPATH}
	RST2HTML="python.exe" "C:\Users\Bojan\Anaconda\Scripts\rst2html.py"
else
	PWEAVE=Pweave
	PPATH=../../:${PYTHONPATH}
	RST2HTML=rst2html5
endif

.PHONY: default
default: docset.zip

%.rst: %.rstw ../../crocodile/simulate.py ../../crocodile/synthesis.py
	PYTHONPATH=${PPATH} ${PWEAVE} $<

%.html: %.rst
	${RST2HTML} $(<:.rstw=.rst)  > $@

.PHONY: clean
clean:
	rm -f ${RSTs} ${HTMLs} docset.zip
	rm -rf figures

.PHONY: rst
rst: $(RSTs)
.PHONY: html
html: $(HTMLs)

docset.zip: html
	zip -r $@ $^ figures

PWs=   wimaging.rstw aa.rstw wkernel.rstw
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


default: docset.zip

%.html: %.rstw ../../crocodile/synthesis.py
	PYTHONPATH=${PPATH} ${PWEAVE} $< ; ${RST2HTML} $(<:.rstw=.rst)  > $@

clean:
	rm -f g${HTMLs} docset.zip

docset.zip: $(HTMLs)
	zip -r $@ $^ figures
